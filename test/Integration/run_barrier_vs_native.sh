#!/bin/bash
# =============================================================================
# run_barrier_vs_native.sh — team barrier call sites, ours against the stock
# compiler, counted at the same stage
#
# run_barrier_stats.sh answers "does the pass do what it claims", inside our
# own pipeline. This answers the different question a reader asks: how many
# barriers are left compared with the compiler people actually use.
#
# The comparison is only worth anything if both sides are counted at the same
# point, so both are taken from LLVM IR *after* -O3:
#
#   clang     clang -fopenmp -O3 -S -emit-llvm
#   ours      the native.sh pipeline through step 6 (opt -S -O3)
#
# Same kernel, same -D$DATASET, same strict-FP flags on both. Counting our
# side in MLIR instead would compare pre-optimisation against post-, and any
# duplication -O3 performs would show up on one side only.
#
# Ours is built twice, with and without --omp-barrier-elim, so the table shows
# both what the pipeline emits on its own and what the pass takes off.
#
# iomp only. Both sides then speak the same ABI (__kmpc_barrier), which is what
# makes a call count meaningful. The gcc/libgomp equivalent is not comparable
# this way: at -O3 gcc emits GOMP_barrier counts that exceed the number of
# work-sharing loops in the source, so the count reflects code duplication as
# much as synchronisation and needs a per-site analysis instead.
#
# Usage:
#   ./run_barrier_vs_native.sh                       # kernels in the suite
#   SUITE=full ./run_barrier_vs_native.sh            # the external PolyBench set
#   ./run_barrier_vs_native.sh path/to/kernel-omp.c  # a single kernel
#   VERBOSE=1 ./run_barrier_vs_native.sh             # let the tools report why
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Nothing is executed and nothing is built for the target.
SKIP_PULP_SDK=1
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

if [ "$RUNTIME" != "iomp" ]; then
    echo "ERROR: this comparison only runs with RUNTIME=iomp — see the header." >&2
    exit 2
fi

# No dumps and no timer: nothing here is run, and both sides must see the same
# defines or they would not be compiling the same program.
POLYBENCH_CFLAGS="$POLYBENCH_ROOT_CFLAGS"

OUTDIR="${OUTDIR:-$PWD/results}/$RUNTIME"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_barrier_vs_native.csv"

ERRSINK="/dev/null"
[ -n "${VERBOSE:-}" ] && ERRSINK="/dev/stderr"

# Our pipeline, native.sh steps 1-6: front-end through opt -O3. Stops before
# llc and the link, which the count does not need.
# $1 = source, $2 = output .ll, $3 = extra mlir-opt-omp flag or ""
build_ours() {
    local src="$1" out="$2" flag="$3"
    local name; name="$(basename "${src%.c}")"

    "$CLANG" -S $CLANG_STRICT_FP $WARN_SUPPRESS \
        -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC" -I"$(dirname "$src")" -I"$INC_OMP" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" -o "$TMPDIR/$name.cir" 2>"$ERRSINK" || return 1
    "$CIR_OPT" "$TMPDIR/$name.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$TMPDIR/$name-s1.mlir" 2>"$ERRSINK" || return 1
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$TMPDIR/$name-s1.mlir"
    "$MLIR_OPT_OMP" --allow-unregistered-dialect \
        --omp-lower-dsl="$RULES" --omp-lower-runtime="$RUNTIME" \
        ${flag:+$flag} \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        "$TMPDIR/$name-s1.mlir" > "$TMPDIR/$name-s2.mlir" 2>"$ERRSINK" || return 1
    "$MLIR_OPT" "$TMPDIR/$name-s2.mlir" \
        --canonicalize --cse --sccp --symbol-dce \
        --loop-invariant-code-motion --canonicalize --cse \
        --convert-arith-to-llvm --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$TMPDIR/$name-s3.mlir" 2>"$ERRSINK" || return 1
    "$MLIR_TRANSLATE" "$TMPDIR/$name-s3.mlir" --mlir-to-llvmir \
        > "$TMPDIR/$name.ll" 2>"$ERRSINK" || return 1
    "$OPT" -S -O3 "$TMPDIR/$name.ll" > "$out" 2>"$ERRSINK" || return 1
}

# Post-O3 the call prints as `tail call void @__kmpc_barrier(...)`, so match the
# callee and count occurrences rather than lines.
count_barriers() {   # $1 = .ll file
    grep -o 'call void @__kmpc_barrier' "$1" | wc -l
}

# How the kernel spells its parallel loop. clang elides the trailing barrier
# only for the combined directive, so this column is what explains the delta.
pragma_form() {   # $1 = source
    if grep -qE '#pragma omp parallel[[:space:]]+for' "$1"; then
        echo "combined"
    else
        echo "split"
    fi
}

if [ $# -gt 0 ]; then
    KERNEL_LIST=("$@")
else
    select_kernels
fi

echo "=== TEAM BARRIERS: OURS vs CLANG (both after -O3) ==="
echo "runtime:   $RUNTIME    dataset: $DATASET"
echo "polybench: $POLYBENCH"
echo "kernels:   ${#KERNEL_LIST[@]}"
echo

echo "kernel,pragma_form,clang,ours_baseline,ours_elim,saved_vs_clang" > "$CSV"
printf '  %-22s %-9s %6s %10s %10s %8s\n' kernel form clang base elim 'vs clang'

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

FAILED=0
T_CLANG=0; T_BASE=0; T_ELIM=0

for k in "${KERNEL_LIST[@]}"; do
    src="$(resolve_src "$k")" || { echo "  SKIP (not found): $k"; continue; }
    name="$(basename "$src" .c)"
    form="$(pragma_form "$src")"

    "$CLANG" -fopenmp -O3 -S -emit-llvm $CLANG_STRICT_FP $WARN_SUPPRESS \
        -I"$INC" -I"$(dirname "$src")" -I"$INC_OMP" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" -o "$TMPDIR/$name-clang.ll" 2>"$ERRSINK" \
        || { echo "  ERROR (clang): $name"; echo "$name,$form,,,," >> "$CSV"; FAILED=1; continue; }

    build_ours "$src" "$TMPDIR/$name-base.ll" "" \
        || { echo "  ERROR (ours, baseline): $name"; echo "$name,$form,,,," >> "$CSV"; FAILED=1; continue; }
    build_ours "$src" "$TMPDIR/$name-elim.ll" "--omp-barrier-elim" \
        || { echo "  ERROR (ours, barrier-elim): $name"; echo "$name,$form,,,," >> "$CSV"; FAILED=1; continue; }

    c="$(count_barriers "$TMPDIR/$name-clang.ll")"
    b="$(count_barriers "$TMPDIR/$name-base.ll")"
    e="$(count_barriers "$TMPDIR/$name-elim.ll")"
    saved=$((c - e))

    T_CLANG=$((T_CLANG + c)); T_BASE=$((T_BASE + b)); T_ELIM=$((T_ELIM + e))
    printf '  %-22s %-9s %6s %10s %10s %8s\n' "$name" "$form" "$c" "$b" "$e" "$saved"
    echo "$name,$form,$c,$b,$e,$saved" >> "$CSV"
done

echo
echo "  clang: $T_CLANG    ours without the pass: $T_BASE    ours with it: $T_ELIM"
if [ "$T_CLANG" -gt 0 ]; then
    awk -v c="$T_CLANG" -v e="$T_ELIM" \
        'BEGIN { printf "  %d fewer than clang (%.1f%%)\n", c - e, 100 * (c - e) / c }'
fi
echo "  Done — $CSV"

# Optional chart: the three columns side by side, clang first. PLOT=true, as in
# the other drivers.
render_barrier_plot "$CSV" "${CSV%.csv}.png"

[ "$FAILED" -ne 0 ] && { echo "  some kernels failed (VERBOSE=1 for the tool output)"; exit 1; }
exit 0
