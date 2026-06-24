#!/bin/bash
# =============================================================================
# run_tasks.sh — end-to-end smoke test for omp.task lowering (libgomp).
#
# Unlike run_correctness.sh / run_performance.sh (PolyBench C kernels through
# the CIR front-end), this driver starts from a hand-written MLIR module so it
# does not depend on the front-end emitting omp.task.  It exercises the part we
# own — the omp.task lowering — and then *runs* the result against real libgomp:
#
#   tasks/task_nested.mlir  (parallel { task { *p = 42 } })
#     -> mlir-opt-omp  (omp-to-omp-lower, omp-outline, omp-lower-plan)
#     -> mlir-opt      (lower to the LLVM dialect)
#     -> mlir-translate-> LLVM IR -> opt -O3 -> llc -> link -lgomp -> run
#
# PASS iff the program prints 42 (the task's write to the shared int is visible
# after the parallel region's implicit barrier).
#
# Tool locations come from common.sh (config.env / env vars), same as the other
# drivers.  This test is libgomp-only.
#
# Usage:
#   ./run_tasks.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This test targets libgomp; set before sourcing so common.sh picks the knobs.
RUNTIME=libgomp
# shellcheck source=common.sh
. "$SCRIPT_DIR/common.sh"

SRC="$SCRIPT_DIR/tasks/task_nested.mlir"
EXPECTED="42"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "=== omp.task END-TO-END (libgomp) ==="
echo "source: $SRC"
echo "rules : $RULES"
echo ""

echo "  [1/5] omp lowering (mlir-opt-omp) ..."
"$MLIR_OPT_OMP" \
    --omp-lower-dsl="$RULES" --omp-lower-runtime=libgomp \
    --omp-to-omp-lower --omp-outline --omp-lower-plan \
    "$SRC" > "$TMP/s2.mlir" || { echo "  ERROR: mlir-opt-omp failed"; exit 1; }

echo "  [2/5] lower to LLVM dialect (mlir-opt) ..."
"$MLIR_OPT" "$TMP/s2.mlir" \
    --convert-arith-to-llvm --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o "$TMP/s3.mlir" || { echo "  ERROR: mlir-opt failed"; exit 1; }

echo "  [3/5] translate to LLVM IR (mlir-translate) ..."
"$MLIR_TRANSLATE" "$TMP/s3.mlir" --mlir-to-llvmir > "$TMP/a.ll" \
    || { echo "  ERROR: mlir-translate failed"; exit 1; }

echo "  [4/5] opt -O3 / llc / link -lgomp ..."
"$OPT" -S -O3 "$TMP/a.ll" > "$TMP/a.opt.ll" || { echo "  ERROR: opt failed"; exit 1; }
"$LLC" -O3 -relocation-model=pic -filetype=obj "$TMP/a.opt.ll" -o "$TMP/a.o" \
    || { echo "  ERROR: llc failed"; exit 1; }
"$CLANG" -no-pie "$TMP/a.o" -lgomp -lm -o "$TMP/task_nested" \
    || { echo "  ERROR: link failed"; exit 1; }

echo "  [5/5] running ..."
got="$(OMP_NUM_THREADS=2 "$TMP/task_nested")" || { echo "  ERROR: program crashed"; exit 1; }
echo "      output: '$got' (expected '$EXPECTED')"
echo ""

if [ "$got" = "$EXPECTED" ]; then
    echo -e "${GREEN}${BOLD}PASS${RESET}"
    exit 0
else
    echo -e "${RED}${BOLD}FAIL${RESET}"
    echo "  --- final LLVM IR ($TMP kept? no) ---"
    exit 1
fi
