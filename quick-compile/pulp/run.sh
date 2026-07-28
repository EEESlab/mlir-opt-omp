#!/bin/bash
# Full pulp pipeline: ../test.c -> CIR -> mlir-opt-omp (pmsis) -> riscv32
# test.o -> PULP-SDK link -> run on gvsoc.
#
# Tool locations come from <repo>/local.env — the same file the Integration
# tests read. It supplies PULP_LLC (a riscv32-capable llc with +xpulpv,
# required) and PULP_SDK_ENV, the GAP SDK environment the link+run step needs so
# make finds $(RULES_DIR)/pmsis_rules.mk. Without the SDK env this stops after
# producing test.o.
set -e
cd "$(dirname "$0")"
OMP_REPO_ROOT="$(cd ../.. && pwd)"
OMP_DEFAULT_TOOL_BIN="$OMP_REPO_ROOT/BUILD"
# shellcheck source=../../scripts/load-local-env.sh
. "$OMP_REPO_ROOT/scripts/load-local-env.sh"

PULP_LLC="${PULP_LLC:?set PULP_LLC (in local.env) to a riscv32-capable llc}"

# The GAP SDK env sets RULES_DIR, which the make below needs. Note RULES_DIR is
# the SDK's, unrelated to RULES (our lowering DSL).
if [ -z "${RULES_DIR:-}" ] && [ -n "${PULP_SDK_ENV:-}" ]; then
    # shellcheck disable=SC1090
    . "$PULP_SDK_ENV" || { echo "ERROR: could not source PULP_SDK_ENV=$PULP_SDK_ENV" >&2; exit 2; }
fi

rm -f *.o *.ll *.cir *.mlir

# C -> CIR -> LLVM-dialect MLIR
"$CLANG" -S -Xclang -fclangir -Xclang -emit-cir -fopenmp -I"$INC_OMP" ../test.c -o test.cir
"$CIR_OPT" test.cir --cir-to-llvm --reconcile-unrealized-casts -o test-s1.mlir
sed -i -E 's/cir\.[^,}]+,? ?//g' test-s1.mlir

# custom OMP lowering (pmsis rules), then down to a riscv32 object
"$MLIR_OPT_OMP" --allow-unregistered-dialect --omp-lower-dsl="$RULES" --omp-lower-runtime=pmsis --omp-to-omp-lower --omp-outline --omp-lower-plan test-s1.mlir > test-s2.mlir
"$MLIR_OPT" test-s2.mlir --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o test-s3.mlir
"$MLIR_TRANSLATE" test-s3.mlir --mlir-to-llvmir > test-s4.ll
"$PULP_LLC" -mtriple=riscv32-unknown-elf -mattr=+m,+c,+xpulpv -relocation-model=pic -filetype=obj test-s4.ll -o test.o

# link + run on gvsoc (PULP-SDK app: pulp_main + cluster_main + adapter + test.o)
if [ -z "${RULES_DIR:-}" ]; then
    echo "test.o built; RULES_DIR unset — set PULP_SDK_ENV or source the GAP SDK environment to link and run"
    exit 0
fi
make clean all run platform=gvsoc
