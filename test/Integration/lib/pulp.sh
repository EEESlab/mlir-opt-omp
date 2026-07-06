#!/bin/bash
# =============================================================================
# pulp.sh — PULP / gvsoc target (RUNTIME=pmsis, TARGET=pulp).
#
# Sourced by common.sh when TARGET=pulp; not meant to be sourced directly.
# Holds everything specific to the PULP-SDK flow: the PULP_* knobs, the SDK
# environment setup, and the build/run helpers.
#
# The pulp target drives an external PULP-SDK application directory: a
# PolyBench-PULP checkout whose Makefile understands
#     make clean all run platform=gvsoc KERNEL_SRC=<kernel.c> [OMP_NATIVE=1|OMP_OPT=1]
# OMP_NATIVE=1 builds the kernel with the SDK's native OpenMP; OMP_OPT=1 links
# the kernel.o we cross-compile through mlir-opt-omp (runtime=pmsis) instead.
#
# Build+run are fused: 'make clean all run' compiles with the PULP-SDK GCC
# toolchain and executes on gvsoc in one shot. A "cell" is one such invocation:
#     ref_seq  make                    (no OpenMP, plain SDK build)
#     ref_par  make OMP_NATIVE=1       (SDK's native OpenMP)
#     opt_seq  kernel.o (no -fopenmp) + make OMP_OPT=1
#     opt_par  kernel.o (-fopenmp)    + make OMP_OPT=1
# Cycle counts are scraped from the 'Cycles = N' line the harness prints.
# =============================================================================

# --- PULP / gvsoc knobs ------------------------------------------------------
# GAP RISC-V GCC toolchain used by the PULP-SDK make (prepended to PATH),
# e.g. .../gap_riscv_toolchain_ubuntu/INSTALL/bin.
PULP_TOOLCHAIN_BIN="${PULP_TOOLCHAIN_BIN:-}"
[ -n "$PULP_TOOLCHAIN_BIN" ] && PATH="$PULP_TOOLCHAIN_BIN:$PATH"

# Optional: a script to source for the GAP SDK environment
# (e.g. $GAP_SDK/configs/gap8_v3.sh). Alternatively source it yourself
# before running the driver. SDK scripts routinely reference unset
# variables, so relax `set -u` around the source.
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
# PolyBench defines baked into the opt kernel.o. Must match what the harness
# Makefile uses for the native (ref) builds, or ref/opt would not be
# comparable: the harness defines both TIME and DUMP_ARRAYS, so we do too
# (on gvsoc everything ends up in the same console log anyway).
PULP_POLYBENCH_DEFS="${PULP_POLYBENCH_DEFS:--DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME}"
PULP_VERBOSE="${PULP_VERBOSE:-0}"            # 1: stream make/gvsoc output

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

# compile_pulp_kernel_obj <src> <outdir> <omp:on|off>
# The CIR/MLIR pipeline for the pulp target: lowers through mlir-opt-omp with
# the pmsis rules and cross-compiles to $PULP_APP_DIR/kernel.o (riscv32), which
# the harness Makefile links when invoked with OMP_OPT=1. The final LLVM IR is
# kept in <outdir> for debugging.
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
        "$src" -o "$tmpdir/$name.cir" || { rm -rf "$tmpdir"; return 1; }

    # 2) CIR -> LLVM dialect MLIR (strip residual cir.* attrs)
    "$CIR_OPT" "$tmpdir/$name.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$tmpdir/$name-s1.mlir" || { rm -rf "$tmpdir"; return 1; }
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$tmpdir/$name-s1.mlir"

    # 3) custom OMP lowering (pmsis rules)
    "$MLIR_OPT_OMP" \
        --allow-unregistered-dialect \
        --omp-lower-dsl="$RULES" \
        --omp-lower-runtime="$RUNTIME" \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        "$tmpdir/$name-s1.mlir" > "$tmpdir/$name-s2.mlir" \
        || { rm -rf "$tmpdir"; return 1; }

    # 4) minimal conversion to the LLVM dialect (no canonicalise/cse pipeline
    #    here — mirrors the proven pulp flow)
    "$MLIR_OPT" "$tmpdir/$name-s2.mlir" \
        --convert-arith-to-llvm --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$tmpdir/$name-s3.mlir" || { rm -rf "$tmpdir"; return 1; }

    # 5) MLIR -> LLVM IR
    "$MLIR_TRANSLATE" "$tmpdir/$name-s3.mlir" --mlir-to-llvmir \
        > "$tmpdir/$name.ll" || { rm -rf "$tmpdir"; return 1; }

    # 6) opt (riscv-capable install), 7) llc -> riscv32 object for the harness
    "$PULP_OPT" $PULP_OPT_FLAGS -S "$tmpdir/$name.ll" -o "$tmpdir/$name.opt.ll" \
        || { rm -rf "$tmpdir"; return 1; }
    "$PULP_LLC" $PULP_LLC_FLAGS -filetype=obj "$tmpdir/$name.opt.ll" \
        -o "$PULP_APP_DIR/kernel.o" || { rm -rf "$tmpdir"; return 1; }

    cp "$tmpdir/$name.opt.ll" "$outdir/${name}_omp-${omp}.ll"
    rm -rf "$tmpdir"
    [ -f "$PULP_APP_DIR/kernel.o" ]
}

# pulp_cell <src> <cell:ref_seq|ref_par|opt_seq|opt_par> <outdir> <logfile>
# Builds and runs one cell on gvsoc (appending to <logfile>) and echoes
# "cycles;bytes" — cycles from the run log, bytes = size of the linked ELF.
# Either field is NA when it could not be determined. Non-zero on build/run
# failure.
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
                || { echo "  ERROR: kernel.o pipeline failed — see $logfile" >&2; return 1; }
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
