#!/bin/bash
# =============================================================================
# run_tasks.sh — end-to-end smoke tests for omp.task lowering.
#
# Two checks, both ending in a real run against the selected runtime
# (libgomp's GOMP_task or libomp's __kmpc_omp_task_alloc/__kmpc_omp_task):
#
#   [1] MLIR  — a hand-written parallel { task { *p = 42 } } module
#               (task_nested.mlir) lowered through mlir-opt-omp and run.
#               Independent of the CIR front-end (does not need clang to emit
#               omp.task), so it always exercises the lowering we own.
#
#   [2] C     — task_smoke.c compiled two ways and compared:
#                 ref : the stock OpenMP compiler ($REF_CC -fopenmp)
#                 opt : clang->CIR->cir-opt->mlir-opt-omp->...->llc->link
#               This is the full front-end path; it depends on ClangIR emitting
#               omp.task.  If your clang-cir lacks task support, [2] fails at the
#               front-end while [1] still passes.
#
#   [3] MLIR  — a hand-written parallel { task; taskwait; read-back } module
#               (taskwait_nested.mlir) where the taskwait is load-bearing: the
#               value the task writes is read back later in the same region,
#               after the taskwait.  Exercises GOMP_taskwait / __kmpc_omp_taskwait
#               end to end.  Front-end independent, like [1].
#
#   [4] C     — taskwait_smoke.c: the same load-bearing taskwait as [3] compiled
#               ref vs opt (like [2]).  Depends on ClangIR emitting omp.taskwait;
#               if it does not, [4] fails at the front-end while [3] still passes.
#
# A test PASSes iff the program prints 42 (the task's write to the shared int is
# visible after the parallel region's implicit barrier).  For [2] the ref and
# opt outputs must also match.
#
# Tool locations and the per-runtime knobs (REF_CC, OPT_FOPENMP,
# OPT_EXTRA_LIBS) come from ../lib/common.sh (config.env / env vars).
#
# Usage:
#   ./run_tasks.sh [libgomp|iomp]     # default: libgomp (or $RUNTIME if set)
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Runtime under test; set before sourcing so common.sh picks the right knobs.
# (pmsis has no task API — common.sh accepts it, but the lowering would fail.)
RUNTIME="${1:-${RUNTIME:-libgomp}}"
# shellcheck source=../lib/common.sh
. "$SCRIPT_DIR/../lib/common.sh"

EXPECTED="42"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

# Where to keep the lowered IR for inspection. Anchored to the Integration
# dir so it lands next to the other drivers' results/ (gitignored, split per
# runtime) no matter where the script is run from.
# One subdir per test; each pipeline stage is written as a numbered file so
# the lowering can be read step by step.
OUTDIR="${OUTDIR:-$SCRIPT_DIR/../results}/$RUNTIME"
DUMP_BASE="$OUTDIR/tasks"
rm -rf "$DUMP_BASE"            # start clean so stale stages don't mislead
mkdir -p "$DUMP_BASE"

# --- shared tail: a (post-front-end) MLIR module -> runnable binary ---------
# Input is in the omp + llvm + func dialects (i.e. the state just before
# mlir-opt-omp); output is an executable linked against $RUNTIME's library.
mlir_to_bin() {
    local in="$1" out="$2" d="${3:-$TMP}"
    mkdir -p "$d"
    local omp_flags=(--allow-unregistered-dialect
        --omp-lower-dsl="$RULES" --omp-lower-runtime="$RUNTIME")

    cp "$in" "$d/01-input.mlir"

    # --allow-unregistered-dialect: CIR-lowered modules carry a dlti.dl_spec
    # attribute (dlti dialect isn't registered in mlir-opt-omp); parse it opaque.
    #
    # Run mlir-opt-omp's own passes incrementally so each stage is captured:
    #   02 omp-to-omp-lower  → omp.* replaced by omp_lower.construct (body intact)
    #   03 + omp-outline     → construct bodies pulled into outlined funcs + calls
    #   04 + omp-lower-plan  → plan attrs lowered to concrete runtime calls
    # Stage 04 is the input to the rest of the pipeline.
    "$MLIR_OPT_OMP" "${omp_flags[@]}" \
        --omp-to-omp-lower \
        "$d/01-input.mlir" > "$d/02-omp-to-omp-lower.mlir" || return 1
    "$MLIR_OPT_OMP" "${omp_flags[@]}" \
        --omp-to-omp-lower --omp-outline \
        "$d/01-input.mlir" > "$d/03-omp-outline.mlir" || return 1
    "$MLIR_OPT_OMP" "${omp_flags[@]}" \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        "$d/01-input.mlir" > "$d/04-omp-lower-plan.mlir" || return 1

    "$MLIR_OPT" "$d/04-omp-lower-plan.mlir" \
        --canonicalize --cse --sccp --symbol-dce \
        --loop-invariant-code-motion --canonicalize --cse \
        --convert-arith-to-llvm --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$d/05-llvm-dialect.mlir" || return 1
    "$MLIR_TRANSLATE" "$d/05-llvm-dialect.mlir" --mlir-to-llvmir \
        > "$d/06-llvmir.ll" || return 1
    "$OPT" -S -O3 "$d/06-llvmir.ll" > "$d/07-opt-O3.ll" || return 1
    "$LLC" -O3 -relocation-model=pic -filetype=obj "$d/07-opt-O3.ll" \
        -o "$d/08-obj.o" || return 1
    # $OPT_FOPENMP/$OPT_EXTRA_LIBS pull in the right runtime library
    # (libgomp: -fopenmp=libgomp -lgomp; iomp: -fopenmp -> libomp).
    "$CLANG" $OPT_FOPENMP -no-pie "$d/08-obj.o" $OPT_EXTRA_LIBS -o "$out" \
        || return 1
}

# --- C -> CIR -> LLVM-dialect MLIR (front-end half of the opt pipeline) -----
c_to_mlir() {
    local src="$1" out="$2" d="${3:-$TMP}"
    mkdir -p "$d"
    local name; name="$(basename "${src%.c}")"
    "$CLANG" -S $CLANG_STRICT_FP $WARN_SUPPRESS \
        -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC_OMP" \
        "$src" -o "$d/00-frontend.cir" || return 1
    "$CIR_OPT" "$d/00-frontend.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$out" || return 1
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$out"
}

echo "=== omp.task END-TO-END ($RUNTIME) ==="
echo "rules: $RULES"
echo ""

# --- [1] hand-written MLIR -------------------------------------------------
echo "── [1] MLIR: parallel { task }"
if mlir_to_bin "$SCRIPT_DIR/task_nested.mlir" "$TMP/mlir_bin" \
        "$DUMP_BASE/mlir_nested"; then
    got="$(OMP_NUM_THREADS=2 "$TMP/mlir_bin" 2>/dev/null || echo '<crash>')"
    echo "     output: '$got' (expected '$EXPECTED')"
    if [ "$got" = "$EXPECTED" ]; then
        echo -e "     ${GREEN}${BOLD}PASS${RESET}"
    else
        echo -e "     ${RED}${BOLD}FAIL${RESET}"; fail=1
    fi
else
    echo -e "     ${RED}${BOLD}ERROR${RESET}: lowering/build failed"; fail=1
fi
echo ""

# --- [2] C through the CIR front-end (ref vs opt) --------------------------
echo "── [2] C: parallel { task }  (ref=$REF_CC vs opt=CIR pipeline)"
CSRC="$SCRIPT_DIR/task_smoke.c"

ref="<n/a>"
if "$REF_CC" -O3 $REF_FP $WARN_SUPPRESS -fopenmp $REF_OMP_INC \
        "$CSRC" -o "$TMP/c_ref"; then
    ref="$(OMP_NUM_THREADS=4 "$TMP/c_ref" 2>/dev/null || echo '<crash>')"
else
    echo -e "     ${RED}ERROR${RESET}: ref ($REF_CC) build failed"; fail=1
fi

opt="<n/a>"
if c_to_mlir "$CSRC" "$TMP/c_s1.mlir" "$DUMP_BASE/c_smoke" \
        && mlir_to_bin "$TMP/c_s1.mlir" "$TMP/c_opt" "$DUMP_BASE/c_smoke"; then
    opt="$(OMP_NUM_THREADS=4 "$TMP/c_opt" 2>/dev/null || echo '<crash>')"
else
    echo -e "     ${RED}ERROR${RESET}: opt (CIR pipeline) build failed"; fail=1
fi

echo "     ref='$ref'  opt='$opt'  (expected '$EXPECTED')"
if [ "$ref" = "$EXPECTED" ] && [ "$opt" = "$EXPECTED" ]; then
    echo -e "     ${GREEN}${BOLD}PASS${RESET}"
else
    echo -e "     ${RED}${BOLD}FAIL${RESET}"; fail=1
fi
echo ""

# --- [3] hand-written MLIR: taskwait is load-bearing -----------------------
# The task writes 42; the region reads it back AFTER omp.taskwait and copies it
# into the printed output.  Without the taskwait the read may observe 0 (a
# deferred task), so this exercises GOMP_taskwait / __kmpc_omp_taskwait end to
# end.  Front-end independent, like [1].
echo "── [3] MLIR: parallel { task; taskwait; read-back }"
if mlir_to_bin "$SCRIPT_DIR/taskwait_nested.mlir" "$TMP/taskwait_bin" \
        "$DUMP_BASE/mlir_taskwait"; then
    got="$(OMP_NUM_THREADS=2 "$TMP/taskwait_bin" 2>/dev/null || echo '<crash>')"
    echo "     output: '$got' (expected '$EXPECTED')"
    if [ "$got" = "$EXPECTED" ]; then
        echo -e "     ${GREEN}${BOLD}PASS${RESET}"
    else
        echo -e "     ${RED}${BOLD}FAIL${RESET}"; fail=1
    fi
else
    echo -e "     ${RED}${BOLD}ERROR${RESET}: lowering/build failed"; fail=1
fi
echo ""

# --- [4] C taskwait through the CIR front-end (ref vs opt) ------------------
# Same load-bearing taskwait as [3], but via the full C front-end path so ref
# (stock compiler) and opt (CIR pipeline) outputs are compared.  Depends on
# ClangIR emitting omp.taskwait; if it does not, this fails at the front-end
# while [3] still covers the lowering.
echo "── [4] C: parallel { task; taskwait; read-back }  (ref=$REF_CC vs opt=CIR pipeline)"
TWSRC="$SCRIPT_DIR/taskwait_smoke.c"

tw_ref="<n/a>"
if "$REF_CC" -O3 $REF_FP $WARN_SUPPRESS -fopenmp $REF_OMP_INC \
        "$TWSRC" -o "$TMP/tw_ref"; then
    tw_ref="$(OMP_NUM_THREADS=4 "$TMP/tw_ref" 2>/dev/null || echo '<crash>')"
else
    echo -e "     ${RED}ERROR${RESET}: ref ($REF_CC) build failed"; fail=1
fi

tw_opt="<n/a>"
if c_to_mlir "$TWSRC" "$TMP/tw_s1.mlir" "$DUMP_BASE/c_taskwait" \
        && mlir_to_bin "$TMP/tw_s1.mlir" "$TMP/tw_opt" "$DUMP_BASE/c_taskwait"; then
    tw_opt="$(OMP_NUM_THREADS=4 "$TMP/tw_opt" 2>/dev/null || echo '<crash>')"
else
    echo -e "     ${RED}ERROR${RESET}: opt (CIR pipeline) build failed"; fail=1
fi

echo "     ref='$tw_ref'  opt='$tw_opt'  (expected '$EXPECTED')"
if [ "$tw_ref" = "$EXPECTED" ] && [ "$tw_opt" = "$EXPECTED" ]; then
    echo -e "     ${GREEN}${BOLD}PASS${RESET}"
else
    echo -e "     ${RED}${BOLD}FAIL${RESET}"; fail=1
fi
echo ""

# --- summary ---------------------------------------------------------------
echo "lowered IR per stage: $DUMP_BASE/<test>/  (01-input … 08-obj)"
if [ "$fail" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}ALL TASK TESTS PASSED${RESET}"
else
    echo -e "${RED}${BOLD}SOME TASK TESTS FAILED${RESET}"
fi
exit "$fail"
