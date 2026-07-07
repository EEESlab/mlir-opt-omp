#!/bin/bash
# =============================================================================
# run_correctness.sh — MLIR OpenMP end-to-end correctness check
#
# Compiles each PolyBench kernel two ways and diffs the dumped arrays:
#   ref  — a stock OpenMP compiler (clang for iomp, gcc for libgomp; for
#          pmsis the PULP-SDK gcc via 'make OMP_NATIVE=1')
#   opt  — the CIR/MLIR pipeline through mlir-opt-omp (selected runtime)
#
# For RUNTIME=pmsis both sides are built and executed on the gvsoc simulator
# through the PolyBench-PULP harness Makefile (PULP_APP_DIR), and the array
# dumps are extracted from the gvsoc console log. Only runs on machines with
# the GAP SDK + gvsoc installed — see config.env.pulp-*.example.
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
#   RUNTIME=pmsis ./run_correctness.sh   # PULP/gvsoc (needs GAP SDK + PULP_APP_DIR)
#   ./run_correctness.sh path/to/kernel-omp.c   # a single kernel
#   DATASET=SMALL_DATASET THREADS=8 ./run_correctness.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

# --- Correctness-specific config -------------------------------------------
THREADS="${THREADS:-16}"
export OMP_NUM_THREADS="$THREADS"
# Results are split per runtime — results/<runtime>/... — so an iomp run only
# replaces a previous iomp run, never a libgomp/pmsis one.
OUTDIR="${OUTDIR:-$PWD/results}/$RUNTIME"

# Correctness dumps the arrays and diffs them.
POLYBENCH_CFLAGS="-DPOLYBENCH_DUMP_ARRAYS $POLYBENCH_ROOT_CFLAGS"

# compare_dumps <name> <ref_dump> <opt_dump> — PASS iff bit-identical.
compare_dumps() {
    local name="$1" ref="$2" opt="$3"
    if diff -q "$ref" "$opt" > /dev/null; then
        echo -e "  ${GREEN}${BOLD}PASS${RESET}"; echo "$name;PASS" >> "$CSV"
    else
        echo -e "  ${RED}${BOLD}FAIL${RESET} — first differences:"
        diff --unified=3 "$ref" "$opt" | head -20
        echo "$name;FAIL" >> "$CSV"
    fi
}

# Pulp/gvsoc path: build+run both sides through the harness Makefile and diff
# the DUMP_ARRAYS sections of the two console logs.
run_kernel_pulp() {
    local src="$1" name="$2" ref_dir="$3" opt_dir="$4"

    echo "  [1/4] ref (make OMP_NATIVE=1, $PULP_PLATFORM)..."
    : > "$ref_dir/run.log"
    if ! pulp_cell "$src" ref_par "$ref_dir" "$ref_dir/run.log" > /dev/null; then
        echo "  ERROR: ref build/run failed"; echo "$name;ERROR" >> "$CSV"; return
    fi

    echo "  [2/4] opt (kernel.o + make OMP_OPT=1, $PULP_PLATFORM)..."
    : > "$opt_dir/run.log"
    if ! pulp_cell "$src" opt_par "$opt_dir" "$opt_dir/run.log" > /dev/null; then
        echo "  ERROR: opt build/run failed"; echo "$name;ERROR" >> "$CSV"; return
    fi

    echo "  [3/4] extracting array dumps..."
    pulp_extract_dump "$ref_dir/run.log" > "$ref_dir/dump.txt"
    pulp_extract_dump "$opt_dir/run.log" > "$opt_dir/dump.txt"
    if [ ! -s "$ref_dir/dump.txt" ] || [ ! -s "$opt_dir/dump.txt" ]; then
        echo "  ERROR: no ==BEGIN DUMP_ARRAYS== section in the gvsoc log —"
        echo "         is -DPOLYBENCH_DUMP_ARRAYS set in the harness Makefile?"
        echo "$name;ERROR" >> "$CSV"; return
    fi

    echo "  [4/4] comparing..."
    compare_dumps "$name" "$ref_dir/dump.txt" "$opt_dir/dump.txt"
    echo ""
}

run_kernel() {
    local kernel="$1" src
    if ! src="$(resolve_src "$kernel")"; then
        echo "  SKIP (not found): $kernel"
        return
    fi
    local name; name="$(basename "${src%-omp.c}")-omp"
    local base="$OUTDIR/$name"
    local ref_dir="$base/ref" opt_dir="$base/opt"
    mkdir -p "$ref_dir" "$opt_dir"

    echo "── $name"
    if [ "$TARGET" = "pulp" ]; then
        run_kernel_pulp "$src" "$name" "$ref_dir" "$opt_dir"
        return
    fi

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
    compare_dumps "$name" "$ref_dir/dump.txt" "$opt_dir/dump.txt"
    echo ""
}

# --- Main ------------------------------------------------------------------
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_correctness.csv"

echo "=== MLIR OpenMP CORRECTNESS CHECK ==="
if [ "$TARGET" = "pulp" ]; then
    echo "runtime: $RUNTIME (pulp/$PULP_PLATFORM)    dataset: $DATASET    suite: $SUITE"
    echo "app dir: $PULP_APP_DIR"
    echo "ref: pulp-sdk gcc ('make OMP_NATIVE=1')    opt: CIR/MLIR kernel.o ('make OMP_OPT=1')"
    echo "polybench: $POLYBENCH"
    echo "rules: $RULES"
    echo "dumps: extracted from the gvsoc console log (POLYBENCH_DUMP_ARRAYS)"
else
    echo "runtime: $RUNTIME    dataset: $DATASET    threads: $THREADS    suite: $SUITE"
    echo "ref cc : $REF_CC"
    echo "polybench: $POLYBENCH"
    echo "rules: $RULES"
    echo "FP mode: strict (-ffp-contract=off, no auto-vectorisation)"
fi
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
