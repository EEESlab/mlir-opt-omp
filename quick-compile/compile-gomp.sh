#!/bin/bash
# Quick pipeline check (libgomp): test.c -> CIR -> mlir-opt-omp -> binary,
# plus a stock-compiler reference (test-ref).
#
# Tool locations come from <repo>/local.env — the same file the Integration
# tests read. Without it the tools are taken from PATH, and mlir-opt-omp from
# ../BUILD.
cd "$(dirname "$0")" || exit 1
OMP_REPO_ROOT="$(cd .. && pwd)"
OMP_DEFAULT_TOOL_BIN="$OMP_REPO_ROOT/BUILD"
# shellcheck source=../scripts/load-local-env.sh
. "$OMP_REPO_ROOT/scripts/load-local-env.sh"

# Narrow on purpose: test.mlir is a checked-in source, not an artefact.
rm -f ./*.o ./*.ll ./*.cir test-s*.mlir test test-ref

"$CLANG" -S -Xclang -fclangir -Xclang -emit-cir -fopenmp -I"$INC_OMP" test.c -o test.cir
"$CIR_OPT" test.cir --cir-to-llvm --reconcile-unrealized-casts -o test-s1.mlir
sed -i -E 's/cir\.[^,}]+,? ?//g' test-s1.mlir
"$MLIR_OPT_OMP" --allow-unregistered-dialect --omp-lower-dsl="$RULES" --omp-lower-runtime=libgomp --omp-to-omp-lower --omp-outline --omp-lower-plan test-s1.mlir > test-s2.mlir
"$MLIR_OPT" test-s2.mlir --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o test-s3.mlir
"$MLIR_TRANSLATE" test-s3.mlir --mlir-to-llvmir > test-s4.ll
"$LLC" -relocation-model=pic -filetype=obj test-s4.ll -o test.o
"$CLANG" -O3 -c main.c
"$CLANG" -O3 -fopenmp test.o main.o -o test

# Reference
"$CLANG" -O3 -fopenmp -I"$INC_OMP" test.c main.c -o test-ref
