#!/bin/bash
# =============================================================================
# run_tasks.sh — end-to-end smoke tests for omp.task lowering (libgomp).
#
# Two checks, both ending in a real run against libgomp:
#
#   [1] MLIR  — a hand-written parallel { task { *p = 42 } } module
#               (task_nested.mlir) lowered through mlir-opt-omp and run.
#               Independent of the CIR front-end (does not need clang to emit
#               omp.task), so it always exercises the lowering we own.
#
#   [2] C     — task_smoke.c compiled two ways and compared:
#                 ref : gcc -fopenmp
#                 opt : clang->CIR->cir-opt->mlir-opt-omp->...->llc->link -lgomp
#               This is the full front-end path; it depends on ClangIR emitting
#               omp.task.  If your clang-cir lacks task support, [2] fails at the
#               front-end while [1] still passes.
#
# A test PASSes iff the program prints 42 (the task's write to the shared int is
# visible after the parallel region's implicit barrier).  For [2] the ref and
# opt outputs must also match.
#
# Tool locations come from ../lib/common.sh (run.env / env vars).  libgomp only.
#
# Usage:
#   ./run_tasks.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This test targets libgomp; set before sourcing so common.sh picks the knobs.
RUNTIME=libgomp
# shellcheck source=../lib/common.sh
. "$SCRIPT_DIR/../lib/common.sh"

EXPECTED="42"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

# Where to keep the lowered IR for inspection. Anchored to the Integration
# dir so it lands next to the other drivers' results/ (gitignored, split per
# runtime — this test is libgomp-only) no matter where the script is run from.
# One subdir per test; each pipeline stage is written as a numbered file so
# the lowering can be read step by step.
OUTDIR="${OUTDIR:-$SCRIPT_DIR/../results}/$RUNTIME"
DUMP_BASE="$OUTDIR/tasks"
rm -rf "$DUMP_BASE"            # start clean so stale stages don't mislead
mkdir -p "$DUMP_BASE"

# --- shared tail: a (post-front-end) MLIR module -> runnable binary ---------
# Input is in the omp + llvm + func dialects (i.e. the state just before
# mlir-opt-omp); output is a libgomp-linked executable.
mlir_to_bin() {
    local in="$1" out="$2" d="${3:-$TMP}"
    mkdir -p "$d"
    local omp_flags=(--allow-unregistered-dialect
        --omp-lower-dsl="$RULES" --omp-lower-runtime=libgomp)

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
    "$CLANG" -no-pie "$d/08-obj.o" -lgomp -lm -o "$out" || return 1
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

echo "=== omp.task END-TO-END (libgomp) ==="
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
echo "── [2] C: parallel { task }  (ref=gcc vs opt=CIR pipeline)"
CSRC="$SCRIPT_DIR/task_smoke.c"

ref="<n/a>"
if "$GCC" -O3 $GCC_STRICT_FP $WARN_SUPPRESS -fopenmp "$CSRC" -o "$TMP/c_ref"; then
    ref="$(OMP_NUM_THREADS=4 "$TMP/c_ref" 2>/dev/null || echo '<crash>')"
else
    echo -e "     ${RED}ERROR${RESET}: ref (gcc) build failed"; fail=1
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

# --- summary ---------------------------------------------------------------
echo "lowered IR per stage: $DUMP_BASE/<test>/  (01-input … 08-obj)"
if [ "$fail" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}ALL TASK TESTS PASSED${RESET}"
else
    echo -e "${RED}${BOLD}SOME TASK TESTS FAILED${RESET}"
fi
exit "$fail"
