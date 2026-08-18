#!/bin/bash
# =============================================================================
# run_barrier_stats.sh — how many team barriers --omp-barrier-elim removes
#
# Runs the front-end half of the pipeline on each kernel (clang -> CIR ->
# LLVM-dialect MLIR, as native.sh does), then the pass alone, and records what
# it reports through --mlir-pass-statistics.
#
# The count is static: barriers gone from the program text, not executions
# saved. One inside a sequential outer loop counts once here and fires once per
# iteration at run time, so the dynamic saving is larger — floyd-warshall being
# the clearest case.
#
# No runtime is selected: the pass reads neither the DSL nor a runtime, so the
# counts hold for all three. What differs is what a barrier costs, which is
# run_performance.sh's job.
#
# Usage:
#   ./run_barrier_stats.sh                       # all kernels in the suite
#   SUITE=full ./run_barrier_stats.sh            # the external PolyBench set
#   ./run_barrier_stats.sh path/to/kernel-omp.c  # a single kernel
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

# Set by each driver, as native.sh expects. Nothing is ever executed here —
# the kernels only get as far as MLIR — so neither the array dumps nor the
# timer of the other drivers are wanted.
POLYBENCH_CFLAGS="$POLYBENCH_ROOT_CFLAGS"

OUTDIR="${OUTDIR:-$PWD/results}"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_barrier_stats.csv"

# A single kernel may be named on the command line, as the other drivers allow.
if [ $# -gt 0 ]; then
    KERNEL_LIST=("$@")
else
    select_kernels
fi

echo "=== BARRIER ELIMINATION STATISTICS ==="
echo "polybench: $POLYBENCH"
echo "kernels:   ${#KERNEL_LIST[@]}"
echo

echo "kernel,wsloops,explicit_removed,implicit_removed" > "$CSV"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

for k in "${KERNEL_LIST[@]}"; do
    src="$(resolve_src "$k")" || { echo "  SKIP (not found): $k"; continue; }
    name="$(basename "$src" .c)"

    # Front-end, mirroring native.sh steps 1 and 2.
    "$CLANG" -S $CLANG_STRICT_FP $WARN_SUPPRESS \
        -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC" -I"$(dirname "$src")" ${INC_OMP:+-I"$INC_OMP"} \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" -o "$tmpdir/$name.cir" 2>/dev/null \
        || { echo "  ERROR (clang): $name"; echo "$name,,," >> "$CSV"; continue; }

    "$CIR_OPT" "$tmpdir/$name.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$tmpdir/$name.mlir" 2>/dev/null \
        || { echo "  ERROR (cir-opt): $name"; echo "$name,,," >> "$CSV"; continue; }
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$tmpdir/$name.mlir"

    # Total work-sharing loops, so removals read as a fraction.
    wsloops="$(grep -c 'omp\.wsloop' "$tmpdir/$name.mlir")"

    stats="$("$MLIR_OPT_OMP" "$tmpdir/$name.mlir" \
        --allow-unregistered-dialect --omp-barrier-elim \
        --mlir-pass-statistics -o /dev/null 2>&1 | grep '(S)')"
    explicit="$(echo "$stats" | grep explicit | awk '{print $2}')"
    implicit="$(echo "$stats" | grep implicit | awk '{print $2}')"

    printf '  %-22s %s wsloops -> %s explicit + %s implicit removed\n' \
        "$name" "$wsloops" "${explicit:-0}" "${implicit:-0}"
    echo "$name,$wsloops,${explicit:-0},${implicit:-0}" >> "$CSV"
done

echo
echo "Done — $CSV"
