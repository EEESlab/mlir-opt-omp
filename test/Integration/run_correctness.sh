#!/bin/bash
# =============================================================================
# run_correctness.sh — MLIR OpenMP end-to-end correctness check
#
# Compiles each PolyBench kernel two ways and diffs the dumped arrays:
#   ref  — a stock OpenMP compiler (clang for iomp, gcc for libgomp)
#   opt  — the CIR/MLIR pipeline through mlir-opt-omp (selected runtime)
#
# A kernel PASSes when the two array dumps are bit-identical. Strict FP flags
# (-ffp-contract=off + no auto-vectorisation, set in common.sh) are required for
# that: without them FMA contraction and reordered reductions make iterative
# kernels (fdtd-2d, jacobi-2d, ...) diverge even though both binaries are
# IEEE-correct.
#
# All shared setup (config, tools, kernel lists, the compile pipeline) lives in
# common.sh. Everything is configurable via environment variables / config.env
# (see config.env.example). Nothing is hard-coded to a particular machine.
#
# Usage:
#   ./run_correctness.sh                 # all kernels, defaults
#   RUNTIME=libgomp ./run_correctness.sh # pick the runtime
#   ./run_correctness.sh path/to/kernel-omp.c   # a single kernel
#   DATASET=SMALL_DATASET THREADS=8 ./run_correctness.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "$SCRIPT_DIR/common.sh"

# --- Correctness-specific config -------------------------------------------
THREADS="${THREADS:-16}"
export OMP_NUM_THREADS="$THREADS"
OUTDIR="${OUTDIR:-$PWD/results}"

# Correctness dumps the arrays and diffs them.
POLYBENCH_CFLAGS="-DPOLYBENCH_DUMP_ARRAYS $POLYBENCH_ROOT_CFLAGS"

run_kernel() {
    local kernel="$1" src
    if ! src="$(resolve_src "$kernel")"; then
        echo "  SKIP (not found): $kernel"
        return
    fi
    local name; name="$(basename "${src%-omp.c}")-omp"
    local base="$OUTDIR/$name"
    local ref_dir="$base/ref$RUNTIME_TAG" opt_dir="$base/opt$RUNTIME_TAG"
    mkdir -p "$ref_dir" "$opt_dir"

    echo "── $name"
    echo "  [1/4] ref..."
    if ! compile_ref "$src" "$ref_dir" "${name}_ref" on; then
        echo "  ERROR: ref compile failed"; echo "$name;ERROR" >> "$CSV"; return
    fi
    echo "  [2/4] opt..."
    if ! compile_opt "$src" "$opt_dir" "${name}_opt" on; then
        echo "  ERROR: opt compile failed"; echo "$name;ERROR" >> "$CSV"; return
    fi

    echo "  [3/4] running..."
    "$ref_dir/${name}_ref" 2> "$ref_dir/dump.txt" > /dev/null || true
    "$opt_dir/${name}_opt" 2> "$opt_dir/dump.txt" > /dev/null || true

    echo "  [4/4] comparing..."
    if diff -q "$ref_dir/dump.txt" "$opt_dir/dump.txt" > /dev/null; then
        echo -e "  ${GREEN}${BOLD}PASS${RESET}"; echo "$name;PASS" >> "$CSV"
    else
        echo -e "  ${RED}${BOLD}FAIL${RESET} — first differences:"
        diff --unified=3 "$ref_dir/dump.txt" "$opt_dir/dump.txt" | head -20
        echo "$name;FAIL" >> "$CSV"
    fi
    echo ""
}

# --- Main ------------------------------------------------------------------
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_correctness$RUNTIME_TAG.csv"

echo "=== MLIR OpenMP CORRECTNESS CHECK ==="
echo "runtime: $RUNTIME    dataset: $DATASET    threads: $THREADS    suite: $SUITE"
echo "ref cc : $REF_CC"
echo "polybench: $POLYBENCH"
echo "rules: $RULES"
echo "FP mode: strict (-ffp-contract=off, no auto-vectorisation)"
echo ""

echo "kernel;result" > "$CSV"

select_kernels

if [ $# -ge 1 ]; then
    run_kernel "$1"
else
    for k in "${KERNEL_LIST[@]}"; do
        run_kernel "$k"
    done
fi

passed=$(grep -c ';PASS$' "$CSV" || true)
failed=$(grep -c ';FAIL$' "$CSV" || true)
errors=$(grep -c ';ERROR$' "$CSV" || true)
total=$((passed + failed + errors))

echo -e "${BOLD}=== SUMMARY ===${RESET}"
echo -e "  ${GREEN}passed: $passed / $total${RESET}"
[ "$failed" -gt 0 ] && echo -e "  ${RED}failed: $failed${RESET}"
[ "$errors" -gt 0 ] && echo -e "  ${YELLOW}errors: $errors${RESET}"
if [ "$failed" -gt 0 ] || [ "$errors" -gt 0 ]; then
    echo ""
    echo "non-passing kernels:"
    grep -E ';(FAIL|ERROR)$' "$CSV" | sed 's/^/  /'
fi
echo ""
echo "Done — $CSV"

# Non-zero exit if anything did not pass, so CI can gate on it.
[ "$failed" -eq 0 ] && [ "$errors" -eq 0 ]
