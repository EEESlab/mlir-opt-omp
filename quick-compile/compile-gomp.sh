#!/bin/bash
# Quick pipeline check (libgomp): test.c -> CIR -> mlir-opt-omp -> binary,
# plus a stock-compiler reference (test-ref). Run from this directory.
rm -f *.o *.ll *.cir *.mlir test test-ref

clang -S -Xclang -fclangir -Xclang -emit-cir -fopenmp -I/usr/lib/gcc/x86_64-linux-gnu/12/include test.c -o test.cir
cir-opt test.cir --cir-to-llvm --reconcile-unrealized-casts -o test-s1.mlir
sed -i -E 's/cir\.[^,}]+,? ?//g' test-s1.mlir
../BUILD/mlir-opt-omp --allow-unregistered-dialect --omp-lower-dsl=../rules.dsl --omp-lower-runtime=libgomp --omp-to-omp-lower --omp-outline --omp-lower-plan test-s1.mlir > test-s2.mlir
mlir-opt test-s2.mlir --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o test-s3.mlir
mlir-translate test-s3.mlir --mlir-to-llvmir > test-s4.ll
llc -relocation-model=pic -filetype=obj test-s4.ll -o test.o
clang -O3 -c main.c
clang -O3 -fopenmp test.o main.o -o test

# Reference
clang -O3 -fopenmp -I/usr/lib/gcc/x86_64-linux-gnu/12/include test.c main.c -o test-ref
