#!/bin/bash
# run_emit_ir.sh — PolyBench kernels from C to a binary, keeping the IR of
# every stage and running the three mlir-opt-omp passes one at a time, so the
# lowering can be read step by step.
#
# Usage:
#   ./run_emit_ir.sh                                        # run.env's KERNELS, else gemm
#   ./run_emit_ir.sh atax                                   # by name
#   ./run_emit_ir.sh gemm jacobi-2d 2mm                     # several
#   ./run_emit_ir.sh all                                    # the whole list
#   ./run_emit_ir.sh linear-algebra/kernels/atax/atax-omp.c # or by path
#   RUNTIME=libgomp ./run_emit_ir.sh atax
#   BARRIER_ELIM=1 ./run_emit_ir.sh                         # adds --omp-barrier-elim as a stage
#
# Leaves results/<runtime>/ir/<kernel>/, starting from a copy of the kernel
# source and then one numbered file per stage.
# pmsis stops after the OpenMP lowering: the rest needs the PULP toolchain.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Nothing here reaches the PULP harness
SKIP_PULP_SDK=1
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

if [ "$BARRIER_ELIM" = "both" ]; then
    echo "ERROR: BARRIER_ELIM=both compares two builds; pick 0 or 1 here." >&2
    exit 2
fi

# --- Which kernels ------------------------------------------------------------
# Resolve every argument up front: a name typed wrong should not surface after
# the first kernel has already been compiled.
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
    # Nothing asked for: take the list run.env already picked, if it picked one.
    if [ -n "${KERNELS:-}" ]; then read -ra ARGS <<< "$KERNELS"; else ARGS=("gemm"); fi
fi
if [ ${#ARGS[@]} -eq 1 ] && [ "${ARGS[0]}" = "all" ]; then
    ARGS=("${ALL_KERNELS[@]}")
fi

SRCS=()
for arg in "${ARGS[@]}"; do
    src="$(resolve_src "$arg")" || {
        echo "ERROR: kernel not found: $arg" >&2
        echo "       Give a name, a path, or 'all'. Names:" >&2
        echo "         $(kernel_names)" >&2
        exit 2
    }
    SRCS+=("$src")
done

# Front-end flags as in compile_opt / compile_pulp_kernel_obj
if [ "$TARGET" = "pulp" ]; then
    # same default as pulp.sh, which is not sourced here
    FRONTEND_FLAGS="-O0 -Xclang -disable-llvm-optzns -DPULP_TARGET \
        ${PULP_POLYBENCH_DEFS:--DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME -DDATA_TYPE_IS_FLOAT}"
else
    POLYBENCH_CFLAGS="-DPOLYBENCH_TIME $POLYBENCH_ROOT_CFLAGS"
    FRONTEND_FLAGS="$CLANG_STRICT_FP $POLYBENCH_CFLAGS"
fi

# --- Helpers ------------------------------------------------------------------
# "a x2, b x1" from a list of names, one per line
count_names() { sort | uniq -c | awk '{printf "%s%s x%s", sep, $2, $1; sep=", "}'; }

# The omp/omp_lower ops still in a stage, and the runtime calls already there.
# omp.private (+ its omp.yield) are module-level privatizer declarations: they
# outlive the lowering unused and disappear in mlir-translate.
summary() {   # summary <file>
    local ops calls
    ops=$(grep -oE '^\s*(%[^=]+=\s*)?"?omp(_lower)?\.[a-z_]+' "$1" \
        | grep -oE 'omp(_lower)?\.[a-z_]+' | count_names)
    calls=$(grep -E 'call' "$1" \
        | grep -oE '@(__kmpc_|GOMP_|ext_pi_|omp_get_)[A-Za-z0-9_]*' \
        | count_names)
    echo "      ops:   ${ops:-none}"
    echo "      calls: ${calls:-none}"
}

# Stages are numbered in the order they run, so BARRIER_ELIM=1 shifts the rest.
begin_stage() {   # begin_stage <name.ext> -> sets CUR to the next numbered file
    CUR="$OUT/$(printf '%02d' "$N")-$1"
    N=$((N + 1))
}
done_stage() {   # done_stage <description>
    printf '  %-28s %s\n' "$(basename "$CUR")" "$1"
    case "$CUR" in *.cir|*.mlir|*.ll) summary "$CUR" ;; esac
    PREV="$CUR"
}
fail() {
    echo -e "${RED}${BOLD}FAILED${RESET} at $(basename "$CUR")" >&2
}

OMP_FLAGS=(--allow-unregistered-dialect
           --omp-lower-dsl="$RULES" --omp-lower-runtime="$RUNTIME")

# --- One kernel, all the stages -----------------------------------------------
emit_ir() {   # emit_ir <src>
    local SRC="$1"
    NAME="$(basename "${SRC%.c}")"
    OUT="${OUTDIR:-$PWD/results}/$RUNTIME$BARRIER_ELIM_DIR_TAG/ir/$NAME"
    rm -rf "$OUT"; mkdir -p "$OUT"
    N=0
    PREV=""

    echo -e "${BOLD}=== $NAME   runtime=$RUNTIME   dataset=$DATASET ===${RESET}"
    echo "    $OUT"
    echo

    # --- C -> CIR -> llvm + omp dialects --------------------------------------
    # The source first, so the directory reads from the C down to the binary.
    begin_stage "$(basename "$SRC")"
    cp "$SRC" "$CUR" || { fail; return 1; }
    done_stage "the kernel, as PolyBench writes it"

    begin_stage cir.cir
    "$CLANG" -S $WARN_SUPPRESS -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC" -I"$(dirname "$SRC")" -I"$INC_OMP" -D"$DATASET" $FRONTEND_FLAGS \
        "$SRC" -o "$CUR" || { fail; return 1; }
    done_stage "ClangIR: the C front-end, omp.* already in place"

    begin_stage llvm-omp.mlir
    "$CIR_OPT" "$PREV" --cir-to-llvm --reconcile-unrealized-casts -o "$CUR" \
        || { fail; return 1; }
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$CUR"
    done_stage "cir-opt: llvm + omp dialects, the input of mlir-opt-omp"

    # --- mlir-opt-omp, one pass at a time -------------------------------------
    if [ -n "$BARRIER_ELIM_FLAG" ]; then
        begin_stage omp-barrier-elim.mlir
        "$MLIR_OPT_OMP" "${OMP_FLAGS[@]}" --omp-barrier-elim "$PREV" -o "$CUR" \
            || { fail; return 1; }
        done_stage "--omp-barrier-elim: redundant team barriers dropped"
    fi

    begin_stage omp-to-omp-lower.mlir
    "$MLIR_OPT_OMP" "${OMP_FLAGS[@]}" --omp-to-omp-lower "$PREV" -o "$CUR" \
        || { fail; return 1; }
    done_stage "--omp-to-omp-lower: omp.* -> omp_lower.construct + DSL plan"

    begin_stage omp-outline.mlir
    "$MLIR_OPT_OMP" "${OMP_FLAGS[@]}" --omp-outline "$PREV" -o "$CUR" \
        || { fail; return 1; }
    done_stage "--omp-outline: bodies outlined, wsloop lowered"

    begin_stage omp-lower-plan.mlir
    "$MLIR_OPT_OMP" "${OMP_FLAGS[@]}" --omp-lower-plan "$PREV" -o "$CUR" \
        || { fail; return 1; }
    done_stage "--omp-lower-plan: plans -> runtime calls"

    if [ "$TARGET" = "pulp" ]; then
        echo
        echo "pmsis: stopped after the OpenMP lowering."
        return 0
    fi

    # --- Generic MLIR/LLVM, as in compile_opt ---------------------------------
    begin_stage llvm-dialect.mlir
    "$MLIR_OPT" "$PREV" \
        --canonicalize --cse --sccp --symbol-dce \
        --loop-invariant-code-motion --canonicalize --cse \
        --convert-arith-to-llvm --convert-func-to-llvm \
        --reconcile-unrealized-casts -o "$CUR" || { fail; return 1; }
    done_stage "mlir-opt: cleanup, everything in the llvm dialect"

    begin_stage llvmir.ll
    "$MLIR_TRANSLATE" "$PREV" --mlir-to-llvmir -o "$CUR" || { fail; return 1; }
    done_stage "mlir-translate: LLVM IR"

    begin_stage opt-O3.ll
    "$OPT" -S -O3 $OPT_STRICT_FP "$PREV" -o "$CUR" || { fail; return 1; }
    done_stage "opt -O3"

    begin_stage obj.o
    "$LLC" -O3 -relocation-model=pic -filetype=obj "$PREV" -o "$CUR" \
        || { fail; return 1; }
    done_stage "llc: object file"

    CUR="$OUT/$NAME"
    "$CLANG" -O3 $CLANG_STRICT_FP -c "$INC/polybench.c" -I"$INC" \
        -D"$DATASET" $POLYBENCH_CFLAGS -o "$OUT/polybench.o" || { fail; return 1; }
    "$CLANG" -O3 $CLANG_STRICT_FP $OPT_FOPENMP -no-pie "$PREV" "$OUT/polybench.o" \
        $OPT_EXTRA_LIBS $POLYBENCH_LFLAGS -o "$CUR" || { fail; return 1; }
    done_stage "linked binary"

    echo
    echo "  run it:  OMP_NUM_THREADS=${THREADS:-4} $CUR"
    return 0
}

# --- Run ----------------------------------------------------------------------
# A kernel that fails does not stop the others: with several of them the
# interesting part is usually which ones got through.
FAILED=()
for src in "${SRCS[@]}"; do
    emit_ir "$src" || FAILED+=("$(basename "${src%.c}")")
    echo
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "${RED}${BOLD}${#FAILED[@]} of ${#SRCS[@]} failed:${RESET} ${FAILED[*]}" >&2
    exit 1
fi
