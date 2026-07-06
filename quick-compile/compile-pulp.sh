#!/bin/bash
# Quick pipeline check (pmsis): test.c -> CIR -> mlir-opt-omp -> riscv32 test.o.
# No link/run: the object is linked by the PULP-SDK harness. Point PULP_LLC at
# a riscv32-capable llc (+xpulpv). Run from this directory.
PULP_LLC="${PULP_LLC:?set PULP_LLC to a riscv32-capable llc}"

rm -f *.o *.ll *.cir *.mlir

clang -S -Xclang -fclangir -Xclang -emit-cir -fopenmp -I/usr/lib/gcc/x86_64-linux-gnu/12/include test.c -o test.cir
cir-opt test.cir --cir-to-llvm --reconcile-unrealized-casts -o test-s1.mlir
sed -i -E 's/cir\.[^,}]+,? ?//g' test-s1.mlir
../BUILD/mlir-opt-omp --allow-unregistered-dialect --omp-lower-dsl=../rules.dsl --omp-lower-runtime=pmsis --omp-to-omp-lower --omp-outline --omp-lower-plan test-s1.mlir > test-s2.mlir
mlir-opt test-s2.mlir --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o test-s3.mlir
mlir-translate test-s3.mlir --mlir-to-llvmir > test-s4.ll
"$PULP_LLC" -mtriple=riscv32-unknown-elf -mattr=+m,+c,+xpulpv -relocation-model=pic -filetype=obj test-s4.ll -o test.o
