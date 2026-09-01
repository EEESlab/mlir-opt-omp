#!/bin/bash
# pulp.sh — the PULP/GAP8 target. Sourced by common.sh only when RUNTIME=pmsis.
#
# Builds and runs each cell through the PolyBench-PULP harness Makefile
# ($PULP_APP_DIR) on the gvsoc simulator: the native side straight from the
# harness, ours by cross-compiling the kernel to $PULP_APP_DIR/kernel.o first.
# Cycles come from the 'Cycles = N' line in the run log.
#
# Needs the GAP SDK, gvsoc, and PULP_APP_DIR — see the PULP section of README.md.

PULP_TOOLCHAIN_BIN="${PULP_TOOLCHAIN_BIN:-}"
[ -n "$PULP_TOOLCHAIN_BIN" ] && PATH="$PULP_TOOLCHAIN_BIN:$PATH"

# Optional: a script to source for the GAP SDK environment
if [ -n "${PULP_SDK_ENV:-}" ]; then
    __had_u=0; case "$-" in *u*) __had_u=1;; esac
    set +u
    # shellcheck disable=SC1090
    . "$PULP_SDK_ENV" || { echo "ERROR: could not source PULP_SDK_ENV=$PULP_SDK_ENV" >&2; exit 2; }
    [ "$__had_u" = 1 ] && set -u
    unset __had_u
fi

# The SDK env and the toolchain may have prepended their own dirs; put the
# LLVM/CIR tools and mlir-opt-omp back in front so ours always win.
[ -n "$LLVM_BIN" ]     && PATH="$LLVM_BIN:$PATH"
[ -n "$OMP_TOOL_BIN" ] && PATH="$OMP_TOOL_BIN:$PATH"
export PATH

PULP_APP_DIR="${PULP_APP_DIR:-}"
if [ -z "$PULP_APP_DIR" ] || [ ! -f "$PULP_APP_DIR/Makefile" ]; then
    echo "ERROR: RUNTIME=pmsis needs PULP_APP_DIR pointing at the PolyBench-PULP" >&2
    echo "       harness (the directory with the PULP-SDK Makefile). This target" >&2
    echo "       only runs on machines with the GAP SDK + gvsoc installed." >&2
    exit 2
fi

PULP_PLATFORM="${PULP_PLATFORM:-gvsoc}"      # gvsoc | board
PULP_MAKE="${PULP_MAKE:-make}"
PULP_MAKE_ARGS="${PULP_MAKE_ARGS:-}"         # extra 'make' args, if any
# riscv32-capable LLVM opt/llc (may be a different install than $OPT/$LLC).
PULP_OPT="${PULP_OPT:-$OPT}"
PULP_LLC="${PULP_LLC:-$LLC}"
PULP_OPT_FLAGS="${PULP_OPT_FLAGS:-}"         # host 'opt' pass; none by default
PULP_LLC_FLAGS="${PULP_LLC_FLAGS:--O3 -mtriple=riscv32-unknown-elf -mattr=+m,+c,+xpulpv -relocation-model=pic}"
# Where the harness Makefile leaves the linked ELF (relative to PULP_APP_DIR).
PULP_BUILD_BIN="${PULP_BUILD_BIN:-BUILD/GAP8_V3/GCC_RISCV_PULPOS/test}"
PULP_POLYBENCH_DEFS="${PULP_POLYBENCH_DEFS:--DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME -DDATA_TYPE_IS_FLOAT}"
PULP_VERBOSE="${PULP_VERBOSE:-0}"            # 1: stream make/gvsoc output
# 1: keep the per-cell scratch directory (the .cir and every .mlir between it
# and the object) instead of deleting it, so a failing step can be rerun by hand.
PULP_KEEP_TMP="${PULP_KEEP_TMP:-0}"
# A cir-opt pass to run on the CIR before --cir-to-llvm. Empty here and set
# per cell by run_unroll.sh, the same way run_performance.sh sets
# BARRIER_ELIM_FLAG for the mlir-opt-omp call below.
CIR_UNROLL_FLAG="${CIR_UNROLL_FLAG:-}"

# --- Helpers -----------------------------------------------------------------

# The harness expects KERNEL_SRC relative to its own directory (./stencils/...).
pulp_kernel_arg() {
    local src="$1"
    case "$src" in
        "$PULP_APP_DIR"/*) echo "./${src#"$PULP_APP_DIR"/}" ;;
        *)                 echo "$src" ;;
    esac
}

# Last 'Cycles = N' in the log (the harness prints it after the kernel run).
pulp_extract_cycles() {
    tr -d '\r' < "$1" \
        | grep -Eo 'Cycles[[:space:]]*=[[:space:]]*[0-9]+' \
        | tail -n 1 | grep -Eo '[0-9]+'
}

# PolyBench's ==BEGIN/END DUMP_ARRAYS== section out of the gvsoc console log.
pulp_extract_dump() {
    tr -d '\r' < "$1" | sed -n '/==BEGIN DUMP_ARRAYS==/,/==END DUMP_ARRAYS==/p'
}

# Drop the scratch directory, or say where it is when asked to keep it.
pulp_drop_tmp() {
    if [ "$PULP_KEEP_TMP" = "1" ]; then
        echo "  intermediates kept in $1"
    else
        rm -rf "$1"
    fi
}

# compile_pulp_kernel_obj <src> <outdir> <omp:on|off>
compile_pulp_kernel_obj() {
    local src="$1" outdir="$2" omp="${3:-on}"
    local name; name="$(basename "${src%.c}")"
    local tmpdir; tmpdir=$(mktemp -d)
    mkdir -p "$outdir"

    local omp_cir=""
    [ "$omp" = "on" ] && omp_cir="-fopenmp"

    echo "  compiling $name (opt, runtime=$RUNTIME, target=pulp, omp=$omp) ..."

    # 1) clang -> CIR. No host optimisation (-O0 + -disable-llvm-optzns): all
    #    optimisation happens in llc for the riscv32 target.
    "$CLANG" -O0 -Xclang -disable-llvm-optzns -S \
        -Xclang -fclangir -Xclang -emit-cir $omp_cir $WARN_SUPPRESS \
        -I"$INC" -I"$(dirname "$src")" -I"$INC_OMP" \
        -DPULP_TARGET -D"$DATASET" $PULP_POLYBENCH_DEFS \
        "$src" -o "$tmpdir/$name.cir" || { pulp_drop_tmp "$tmpdir"; return 1; }

    # 2) CIR -> LLVM dialect MLIR (strip residual cir.* attrs). A CIR-level
    #    pass, when one was asked for, runs here: on the CIR, while it is
    #    still there to work on.
    #    --mlir-disable-threading: with the thread pool on, the unroll pass
    #    fails on a different kernel each run, always as a segfault inside
    #    the parallel verifier. Single-threaded the run is reproducible, and
    #    a pass that really does produce bad IR then reports it instead of
    #    crashing. Costs nothing measurable on modules this size, and it
    #    applies to both cells so the two stay comparable.
    "$CIR_OPT" "$tmpdir/$name.cir" \
        --mlir-disable-threading \
        ${CIR_UNROLL_FLAG:+$CIR_UNROLL_FLAG} \
        --cir-to-llvm --reconcile-unrealized-casts \
        -o "$tmpdir/$name-s1.mlir" || { pulp_drop_tmp "$tmpdir"; return 1; }
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$tmpdir/$name-s1.mlir"

    # 3) custom OMP lowering (pmsis rules)
    "$MLIR_OPT_OMP" \
        --allow-unregistered-dialect \
        --omp-lower-dsl="$RULES" \
        --omp-lower-runtime="$RUNTIME" \
        ${BARRIER_ELIM_FLAG:+$BARRIER_ELIM_FLAG} \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        "$tmpdir/$name-s1.mlir" > "$tmpdir/$name-s2.mlir" \
        || { pulp_drop_tmp "$tmpdir"; return 1; }

    # 4) minimal conversion to the LLVM dialect (no canonicalise/cse pipeline
    #    here — mirrors the proven pulp flow)
    "$MLIR_OPT" "$tmpdir/$name-s2.mlir" \
        --convert-arith-to-llvm --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$tmpdir/$name-s3.mlir" || { pulp_drop_tmp "$tmpdir"; return 1; }

    # 5) MLIR -> LLVM IR
    "$MLIR_TRANSLATE" "$tmpdir/$name-s3.mlir" --mlir-to-llvmir \
        > "$tmpdir/$name.ll" || { pulp_drop_tmp "$tmpdir"; return 1; }

    # 6) opt (riscv-capable install), 7) llc -> riscv32 object for the harness
    "$PULP_OPT" $PULP_OPT_FLAGS -S "$tmpdir/$name.ll" -o "$tmpdir/$name.opt.ll" \
        || { pulp_drop_tmp "$tmpdir"; return 1; }
    "$PULP_LLC" $PULP_LLC_FLAGS -filetype=obj "$tmpdir/$name.opt.ll" \
        -o "$PULP_APP_DIR/kernel.o" || { pulp_drop_tmp "$tmpdir"; return 1; }

    cp "$tmpdir/$name.opt.ll" \
        "$outdir/${name}_omp-${omp}${CIR_UNROLL_FLAG:+_unrolled}.ll"
    pulp_drop_tmp "$tmpdir"
    [ -f "$PULP_APP_DIR/kernel.o" ]
}

pulp_cell() {
    local src="$1" cell="$2" outdir="$3" logfile="$4"
    local karg; karg="$(pulp_kernel_arg "$src")"
    local -a mk=(clean all run "platform=$PULP_PLATFORM" "KERNEL_SRC=$karg")

    case "$cell" in
        ref_seq) ;;
        ref_par) mk+=(OMP_NATIVE=1) ;;
        opt_seq|opt_par)
            local omp="off"; [ "$cell" = "opt_par" ] && omp="on"
            compile_pulp_kernel_obj "$src" "$outdir" "$omp" >> "$logfile" 2>&1 \
                || { echo "  ERROR: kernel.o pipeline failed — last lines of $logfile:" >&2
                     tail -n 8 "$logfile" >&2; return 1; }
            mk+=(OMP_OPT=1)
            ;;
        *) echo "pulp_cell: bad cell '$cell'" >&2; return 1 ;;
    esac

    if [ "$PULP_VERBOSE" = "1" ]; then
        ( cd "$PULP_APP_DIR" && "$PULP_MAKE" "${mk[@]}" $PULP_MAKE_ARGS ) 2>&1 \
            | tee -a "$logfile" >&2 \
            || { echo "  ERROR: make/gvsoc failed — see $logfile" >&2; return 1; }
    else
        ( cd "$PULP_APP_DIR" && "$PULP_MAKE" "${mk[@]}" $PULP_MAKE_ARGS ) >> "$logfile" 2>&1 \
            || { echo "  ERROR: make/gvsoc failed — last lines of $logfile:" >&2
                 tail -n 8 "$logfile" >&2; return 1; }
    fi

    local cycles bytes
    cycles="$(pulp_extract_cycles "$logfile")"
    [ -n "$cycles" ] || cycles="NA"
    bytes="$(stat -c%s "$PULP_APP_DIR/$PULP_BUILD_BIN" 2>/dev/null)" || bytes="NA"
    [ -n "$bytes" ] || bytes="NA"
    printf '%s;%s' "$cycles" "$bytes"
}
