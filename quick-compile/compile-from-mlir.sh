#!/bin/bash
# Quick check of the passes on their own: MLIR -> mlir-opt-omp -> binary, with
# no C front-end in the way. The input is already in the dialects the tool
# consumes, so this path needs neither ClangIR nor cir-opt — a stock LLVM/MLIR
# install plus a built mlir-opt-omp is enough.
#
#   ./compile-from-mlir.sh [libgomp|iomp] [input.mlir]
#
# Defaults to libgomp and test.mlir, the MLIR twin of test.c: it defines the
# same @add, so it links against the same main.c and must print the same
# numbers. Any module in the omp + llvm dialects works, including the
# test-s1.mlir that compile-gomp.sh / compile-iomp.sh leave behind — feeding
# that back in runs the second half of the C pipeline on its own.
#
# Tool locations come from <repo>/local.env — the same file the Integration
# tests read. Without it the tools are taken from PATH, and mlir-opt-omp from
# ../BUILD.
cd "$(dirname "$0")" || exit 1
OMP_REPO_ROOT="$(cd .. && pwd)"
OMP_DEFAULT_TOOL_BIN="$OMP_REPO_ROOT/BUILD"
# shellcheck source=../scripts/load-local-env.sh
. "$OMP_REPO_ROOT/scripts/load-local-env.sh"

RUNTIME="${1:-libgomp}"
INPUT="${2:-test.mlir}"

case "$RUNTIME" in
    libgomp) LINK_FLAGS=(-lgomp -lm) ;;
    iomp)    LINK_FLAGS=(-fopenmp -lm) ;;
    *) echo "usage: $0 [libgomp|iomp] [input.mlir]" >&2; exit 2 ;;
esac
if [ ! -f "$INPUT" ]; then
    echo "$0: no such file: $INPUT" >&2; exit 2
fi

rm -f ./*.o ./*.ll test-m*.mlir test

# The three passes in one go, then the standard MLIR/LLVM tail.
# --allow-unregistered-dialect is not needed by test.mlir; it is there so a
# module coming from cir-opt (which carries a dlti.dl_spec attribute) parses too.
set -e
"$MLIR_OPT_OMP" --allow-unregistered-dialect \
    --omp-lower-dsl="$RULES" --omp-lower-runtime="$RUNTIME" \
    --omp-to-omp-lower --omp-outline --omp-lower-plan \
    "$INPUT" > test-m1.mlir
"$MLIR_OPT" test-m1.mlir --canonicalize --cse --sccp --symbol-dce \
    --loop-invariant-code-motion --canonicalize --cse \
    --convert-arith-to-llvm --convert-func-to-llvm \
    --reconcile-unrealized-casts -o test-m2.mlir
"$MLIR_TRANSLATE" test-m2.mlir --mlir-to-llvmir > test-m3.ll
"$OPT" -O3 test-m3.ll -S -o test-m3.opt.ll
"$LLC" -O3 -relocation-model=pic -filetype=obj test-m3.opt.ll -o test.o
"$CLANG" -O3 -c main.c -o main.o
"$CLANG" -no-pie test.o main.o "${LINK_FLAGS[@]}" -o test
set +e

# With the bundled test.mlir the answer is known, so say whether it is right
# instead of leaving it to the eye. For any other input just show the output.
echo ""
got="$(./test 2>/dev/null || echo '<crash>')"
if [ "$INPUT" = "test.mlir" ]; then
    want="$(printf '%d\n' 11 22 33 44 55 66 77 88 99 110)"
    if [ "$got" = "$want" ]; then
        echo "PASS ($RUNTIME): ./test printed 11..110"
    else
        echo "FAIL ($RUNTIME): unexpected output from ./test"
        echo "--- got ---"; echo "$got"
        echo "--- want ---"; echo "$want"
        exit 1
    fi
else
    echo "$RUNTIME: ./test printed"
    echo "$got"
fi
