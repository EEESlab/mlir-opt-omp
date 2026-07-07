#!/bin/bash
# Full pulp pipeline: ../test.c -> CIR -> mlir-opt-omp (pmsis) -> riscv32
# test.o -> PULP-SDK link -> run on gvsoc.
# Needs PULP_LLC pointing at a riscv32-capable llc (+xpulpv) and, for the
# link+run step, the GAP SDK environment sourced so make finds
# $(RULES_DIR)/pmsis_rules.mk (e.g. `source $GAP_SDK/configs/gap8_v3.sh`).
# Without the SDK env it stops after producing test.o.
set -e
cd "$(dirname "$0")"

PULP_LLC="${PULP_LLC:?set PULP_LLC to a riscv32-capable llc}"

rm -f *.o *.ll *.cir *.mlir

# C -> CIR -> LLVM-dialect MLIR
clang -S -Xclang -fclangir -Xclang -emit-cir -fopenmp -I/usr/lib/gcc/x86_64-linux-gnu/12/include ../test.c -o test.cir
cir-opt test.cir --cir-to-llvm --reconcile-unrealized-casts -o test-s1.mlir
sed -i -E 's/cir\.[^,}]+,? ?//g' test-s1.mlir

# custom OMP lowering (pmsis rules), then down to a riscv32 object
../../BUILD/mlir-opt-omp --allow-unregistered-dialect --omp-lower-dsl=../../rules.dsl --omp-lower-runtime=pmsis --omp-to-omp-lower --omp-outline --omp-lower-plan test-s1.mlir > test-s2.mlir
mlir-opt test-s2.mlir --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o test-s3.mlir
mlir-translate test-s3.mlir --mlir-to-llvmir > test-s4.ll
"$PULP_LLC" -mtriple=riscv32-unknown-elf -mattr=+m,+c,+xpulpv -relocation-model=pic -filetype=obj test-s4.ll -o test.o

# link + run on gvsoc (PULP-SDK app: pulp_main + cluster_main + adapter + test.o)
if [ -z "${RULES_DIR:-}" ]; then
    echo "test.o built; RULES_DIR unset — source the GAP SDK environment to link and run"
    exit 0
fi
make clean all run platform=gvsoc
