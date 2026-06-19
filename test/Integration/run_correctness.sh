#!/bin/bash
# =============================================================================
# run_correctness.sh — MLIR OpenMP end-to-end correctness check
#
# Compiles each PolyBench kernel two ways and diffs the dumped arrays:
#   ref  — a stock OpenMP compiler (clang for iomp, gcc for libgomp)
#   opt  — the CIR/MLIR pipeline through mlir-opt-omp (selected runtime)
#
# A kernel PASSes when the two array dumps are bit-identical. Strict FP flags
# (-ffp-contract=off + no auto-vectorisation) are required for that: without
# them FMA contraction and reordered reductions make iterative kernels (fdtd-2d,
# jacobi-2d, ...) diverge even though both binaries are IEEE-correct.
#
# Everything is configurable via environment variables (see config.env.example).
# Nothing is hard-coded to a particular machine: set the vars, or drop a
# `config.env` next to this script, and it runs anywhere.
#
# Usage:
#   ./run_correctness.sh                 # all kernels, defaults
#   RUNTIME=libgomp ./run_correctness.sh # pick the runtime
#   ./run_correctness.sh path/to/kernel-omp.c   # a single kernel
#   DATASET=SMALL_DATASET THREADS=8 ./run_correctness.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colours (disabled when stdout is not a terminal).
if [ -t 1 ]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
    BOLD='\033[1m'; RESET='\033[0m'
else
    GREEN=''; RED=''; YELLOW=''; BOLD=''; RESET=''
fi

# Optional per-machine config: copy config.env.example -> config.env and edit.
# Values already set in the environment take precedence over the file.
if [ -f "$SCRIPT_DIR/config.env" ]; then
    # shellcheck disable=SC1091
    set -a; . "$SCRIPT_DIR/config.env"; set +a
fi

# --- Tool locations --------------------------------------------------------
# If LLVM_BIN / OMP_TOOL_BIN point at directories, they are prepended to PATH
# so the bare tool names below resolve to the right build.
LLVM_BIN="${LLVM_BIN:-}"
OMP_TOOL_BIN="${OMP_TOOL_BIN:-}"
[ -n "$LLVM_BIN" ]     && PATH="$LLVM_BIN:$PATH"
[ -n "$OMP_TOOL_BIN" ] && PATH="$OMP_TOOL_BIN:$PATH"
export PATH

CLANG="${CLANG:-clang}"
GCC="${GCC:-gcc}"
OPT="${OPT:-opt}"
LLC="${LLC:-llc}"
CIR_OPT="${CIR_OPT:-cir-opt}"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
MLIR_OPT_OMP="${MLIR_OPT_OMP:-mlir-opt-omp}"

# --- Paths -----------------------------------------------------------------
# Defaults to the kernels vendored in this repo, so the test is self-contained.
# Point POLYBENCH at a full PolyBench/OMP checkout (and use SUITE=full) to run
# the whole benchmark set.
POLYBENCH="${POLYBENCH:-$SCRIPT_DIR/kernels}"
INC="${POLYBENCH_UTIL:-$POLYBENCH/utilities}"
RULES="${RULES:-$SCRIPT_DIR/../../rules.dsl}"
# OpenMP headers used by the clang->CIR front-end. Adjust to your GCC version.
INC_OMP="${INC_OMP:-/usr/lib/gcc/x86_64-linux-gnu/13/include}"
OUTDIR="${OUTDIR:-$PWD/results}"

# --- Run parameters --------------------------------------------------------
RUNTIME="${RUNTIME:-iomp}"           # iomp | libgomp
DATASET="${DATASET:-MINI_DATASET}"
THREADS="${THREADS:-16}"
SUITE="${SUITE:-bundled}"            # bundled | full

export OMP_NUM_THREADS="$THREADS"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"

POLYBENCH_CFLAGS="-DPOLYBENCH_DUMP_ARRAYS"

# Running as root needs the FIFO scheduler define + an explicit libc link
# (PolyBench convention). Honoured automatically; override POLYBENCH_LFLAGS to
# disable.
POLYBENCH_LFLAGS="${POLYBENCH_LFLAGS:-}"
if [ "$(id -u)" -eq 0 ]; then
    POLYBENCH_CFLAGS="$POLYBENCH_CFLAGS -DPOLYBENCH_LINUX_FIFO_SCHEDULER"
    POLYBENCH_LFLAGS="$POLYBENCH_LFLAGS -lc"
fi

# Strict FP flags — keep ref and opt in sync.
CLANG_STRICT_FP="${CLANG_STRICT_FP:--ffp-contract=off -fno-vectorize -fno-slp-vectorize}"
GCC_STRICT_FP="${GCC_STRICT_FP:--ffp-contract=off -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize}"

# --- Per-runtime knobs -----------------------------------------------------
case "$RUNTIME" in
    iomp)
        REF_CC="$CLANG"
        REF_FP="$CLANG_STRICT_FP"
        REF_OMP_INC="-I$INC_OMP"
        OPT_FOPENMP="-fopenmp"
        OPT_EXTRA_LIBS="-lm"
        ;;
    libgomp)
        REF_CC="$GCC"
        REF_FP="$GCC_STRICT_FP"
        REF_OMP_INC=""
        OPT_FOPENMP="-fopenmp=libgomp"
        OPT_EXTRA_LIBS="-lm -lgomp"
        ;;
    *)
        echo "ERROR: unknown RUNTIME='$RUNTIME' (expected iomp or libgomp)" >&2
        exit 2
        ;;
esac

# Kernels bundled in this repo (paths relative to $POLYBENCH).
BUNDLED_KERNELS=(
    "linear-algebra/blas/gemm/gemm-omp.c"
    "linear-algebra/kernels/atax/atax-omp.c"
)

# The full PolyBench/OMP suite (paths relative to $POLYBENCH). Used with
# SUITE=full against an external checkout.
ALL_KERNELS=(
    "datamining/covariance/covariance-omp.c"
    "datamining/correlation/correlation-omp.c"
    "stencils/jacobi-1d/jacobi-1d-omp.c"
    "stencils/heat-3d/heat-3d-omp.c"
    "stencils/fdtd-2d/fdtd-2d-omp.c"
    "stencils/jacobi-2d/jacobi-2d-omp.c"
    "stencils/adi/adi-omp.c"
    "linear-algebra/blas/gemm/gemm-omp.c"
    "linear-algebra/blas/gesummv/gesummv-omp.c"
    "linear-algebra/blas/trmm/trmm-omp.c"
    "linear-algebra/blas/gemver/gemver-omp.c"
    "linear-algebra/blas/syrk/syrk-omp.c"
    "linear-algebra/blas/syr2k/syr2k-omp.c"
    "linear-algebra/blas/symm/symm-omp.c"
    "linear-algebra/solvers/gramschmidt/gramschmidt-omp.c"
    "linear-algebra/solvers/lu/lu-omp.c"
    "linear-algebra/solvers/cholesky/cholesky-omp.c"
    "linear-algebra/solvers/ludcmp/ludcmp-omp.c"
    "linear-algebra/solvers/trisolv/trisolv-omp.c"
    "linear-algebra/solvers/durbin/durbin-omp.c"
    "linear-algebra/kernels/mvt/mvt-omp.c"
    "linear-algebra/kernels/atax/atax-omp.c"
    "linear-algebra/kernels/doitgen/doitgen-omp.c"
    "linear-algebra/kernels/bicg/bicg-omp.c"
    "linear-algebra/kernels/2mm/2mm-omp.c"
    "linear-algebra/kernels/3mm/3mm-omp.c"
    "medley/floyd-warshall/floyd-warshall-omp.c"
    "medley/deriche/deriche-omp.c"
    "medley/nussinov/nussinov-omp.c"
    "stencils/seidel-2d/seidel-2d-omp.c"
)

# Resolve a kernel argument: accept an absolute/relative path or a path under
# $POLYBENCH.
resolve_src() {
    local k="$1"
    if [ -f "$k" ]; then echo "$k"; return 0; fi
    if [ -f "$POLYBENCH/$k" ]; then echo "$POLYBENCH/$k"; return 0; fi
    return 1
}

# --- opt: the CIR/MLIR pipeline --------------------------------------------
compile_opt() {
    local src="$1" outdir="$2" binname="$3"
    local name; name="$(basename "${src%.c}")"
    local tmpdir; tmpdir=$(mktemp -d)

    echo "  compiling $name (opt, runtime=$RUNTIME) ..."

    # 1) clang -> CIR
    "$CLANG" -S $CLANG_STRICT_FP \
        -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC" -I"$(dirname "$src")" -I"$INC_OMP" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" -o "$tmpdir/$name.cir" || { rm -rf "$tmpdir"; return 1; }

    # 2) CIR -> LLVM dialect MLIR (strip residual cir.* attrs)
    "$CIR_OPT" "$tmpdir/$name.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$tmpdir/$name-s1.mlir" || { rm -rf "$tmpdir"; return 1; }
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$tmpdir/$name-s1.mlir"

    # 3) custom OMP lowering
    "$MLIR_OPT_OMP" \
        --allow-unregistered-dialect \
        --omp-lower-dsl="$RULES" \
        --omp-lower-runtime="$RUNTIME" \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        "$tmpdir/$name-s1.mlir" > "$tmpdir/$name-s2.mlir" \
        || { rm -rf "$tmpdir"; return 1; }

    # 4) MLIR opt + lowering to LLVM dialect
    "$MLIR_OPT" "$tmpdir/$name-s2.mlir" \
        --canonicalize --cse --sccp --symbol-dce \
        --loop-invariant-code-motion --canonicalize --cse \
        --convert-arith-to-llvm --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$tmpdir/$name-s3.mlir" || { rm -rf "$tmpdir"; return 1; }

    # 5) MLIR -> LLVM IR
    "$MLIR_TRANSLATE" "$tmpdir/$name-s3.mlir" --mlir-to-llvmir \
        > "$tmpdir/$name.ll" || { rm -rf "$tmpdir"; return 1; }

    # 6) opt -O3, 7) llc -> object
    "$OPT" -S -O3 "$tmpdir/$name.ll" > "$tmpdir/$name.opt.ll" \
        || { rm -rf "$tmpdir"; return 1; }
    "$LLC" -O3 -relocation-model=pic -filetype=obj "$tmpdir/$name.opt.ll" \
        -o "$tmpdir/$name.o" || { rm -rf "$tmpdir"; return 1; }

    # 8) polybench support object
    "$CLANG" -O3 $CLANG_STRICT_FP \
        -c "$INC/polybench.c" -I"$INC" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        -o "$tmpdir/polybench.o" || { rm -rf "$tmpdir"; return 1; }

    # 9) link
    "$CLANG" -O3 $CLANG_STRICT_FP \
        $OPT_FOPENMP -no-pie \
        "$tmpdir/$name.o" "$tmpdir/polybench.o" \
        $OPT_EXTRA_LIBS $POLYBENCH_LFLAGS -o "$outdir/$binname" \
        || { rm -rf "$tmpdir"; return 1; }

    cp "$tmpdir/$name.opt.ll" "$outdir/$binname.ll"   # keep final IR for debugging
    rm -rf "$tmpdir"
    [ -f "$outdir/$binname" ]
}

# --- ref: stock OpenMP compiler --------------------------------------------
compile_ref() {
    local src="$1" outdir="$2" binname="$3"
    mkdir -p "$outdir"
    echo "  compiling $(basename "${src%.c}") (ref, $REF_CC) ..."

    "$REF_CC" -O3 $REF_FP -fopenmp \
        -I"$INC" -I"$(dirname "$src")" $REF_OMP_INC \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" "$INC/polybench.c" \
        -lm $POLYBENCH_LFLAGS -o "$outdir/$binname" || return 1
    [ -f "$outdir/$binname" ]
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
    echo "  [1/4] ref..."
    if ! compile_ref "$src" "$ref_dir" "${name}_ref"; then
        echo "  ERROR: ref compile failed"; echo "$name;ERROR" >> "$CSV"; return
    fi
    echo "  [2/4] opt..."
    if ! compile_opt "$src" "$opt_dir" "${name}_opt"; then
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
CSV="$OUTDIR/results_correctness.csv"

echo "=== MLIR OpenMP CORRECTNESS CHECK ==="
echo "runtime: $RUNTIME    dataset: $DATASET    threads: $THREADS    suite: $SUITE"
echo "ref cc : $REF_CC"
echo "polybench: $POLYBENCH"
echo "rules: $RULES"
echo "FP mode: strict (-ffp-contract=off, no auto-vectorisation)"
echo ""

echo "kernel;result" > "$CSV"

# Select the kernel list: an explicit KERNELS override wins, otherwise SUITE
# picks the bundled set or the full suite.
if [ -n "${KERNELS:-}" ]; then
    read -ra KERNEL_LIST <<< "$KERNELS"
elif [ "$SUITE" = "full" ]; then
    KERNEL_LIST=("${ALL_KERNELS[@]}")
else
    KERNEL_LIST=("${BUNDLED_KERNELS[@]}")
fi

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
