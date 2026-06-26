#!/bin/bash
# =============================================================================
# common.sh — shared setup for the Integration test drivers.
#
# Sourced by both run_correctness.sh (diff array dumps) and run_performance.sh
# (time + compare). Owns everything the two layers have in common so the
# compile pipeline lives in exactly one place:
#
#   - config.env loading + tool/path resolution
#   - per-runtime knobs (iomp/clang vs libgomp/gcc)
#   - strict-FP flags, root/FIFO handling
#   - the kernel lists (bundled + full PolyBench suite)
#   - compile_ref() / compile_opt()  — the C -> binary pipelines
#
# The caller sets POLYBENCH_CFLAGS *before* invoking the compile functions:
#   correctness -> -DPOLYBENCH_DUMP_ARRAYS
#   performance -> -DPOLYBENCH_TIME -DPOLYBENCH_CYCLE_ACCURATE_TIMER
# common.sh appends the root-only define automatically (see below).
#
# Nothing here is hard-coded to a machine: set env vars, or drop a config.env
# next to the scripts, and it runs anywhere.
# =============================================================================

# Resolve the directory of this file (works regardless of the caller's CWD).
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Colours (disabled when stdout is not a terminal) ----------------------
if [ -t 1 ]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'
    YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'
else
    GREEN=''; RED=''; CYAN=''; YELLOW=''; BOLD=''; RESET=''
fi

# --- Optional per-machine config -------------------------------------------
# config.env is git-ignored. Precedence: inline env (FOO=bar ./script) wins,
# then config.env, then the built-in defaults below.
#
# config.env uses plain assignments (FOO="..."), which would clobber an inline
# override when sourced. So we snapshot any knob already set in the environment,
# source config.env, then re-apply the snapshot — making inline overrides win.
if [ -f "$COMMON_DIR/config.env" ]; then
    __preset=""
    for __v in LLVM_BIN OMP_TOOL_BIN CLANG GCC OPT LLC CIR_OPT MLIR_OPT \
               MLIR_TRANSLATE MLIR_OPT_OMP POLYBENCH POLYBENCH_UTIL RULES \
               INC_OMP OUTDIR RUNTIME DATASET THREADS SUITE KERNELS REPS \
               VARIANCE_ACCEPTED POLYBENCH_LFLAGS CLANG_STRICT_FP GCC_STRICT_FP \
               OMP_PLACES OMP_PROC_BIND; do
        [ -n "${!__v+x}" ] && __preset="$__preset $__v=$(printf '%q' "${!__v}")"
    done
    # shellcheck disable=SC1091
    set -a; . "$COMMON_DIR/config.env"; set +a
    [ -n "$__preset" ] && eval "$__preset"   # inline overrides reclaim priority
    unset __preset __v
fi

# --- Tool locations --------------------------------------------------------
# LLVM_BIN / OMP_TOOL_BIN, if set, are prepended to PATH so the bare tool
# names below resolve to the right build.
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
# Defaults to the kernels vendored in this repo, so the tests are
# self-contained. Point POLYBENCH at a full PolyBench/OMP checkout (and use
# SUITE=full) to run the whole benchmark set.
POLYBENCH="${POLYBENCH:-$COMMON_DIR/kernels}"
INC="${POLYBENCH_UTIL:-$POLYBENCH/utilities}"
RULES="${RULES:-$COMMON_DIR/../../rules.dsl}"
# OpenMP headers used by the clang->CIR front-end. Adjust to your GCC version.
INC_OMP="${INC_OMP:-/usr/lib/gcc/x86_64-linux-gnu/13/include}"
# Serial OpenMP stubs, linked into the sequential builds (see compile_*).
OMP_STUBS_SRC="$COMMON_DIR/omp_stubs.c"

# --- Run parameters --------------------------------------------------------
# DATASET_DEFAULT lets a driver pick its own default (e.g. LARGE for perf)
# while still letting the user/config.env override via DATASET.
RUNTIME="${RUNTIME:-iomp}"           # iomp | libgomp
DATASET="${DATASET:-${DATASET_DEFAULT:-MINI_DATASET}}"
SUITE="${SUITE:-bundled}"            # bundled | full

export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"

# --- Root / FIFO scheduler handling ----------------------------------------
# Running as root lets PolyBench use the FIFO scheduler for lower variance, but
# it needs the define + an explicit libc link. Exposed as POLYBENCH_ROOT_CFLAGS
# so each driver can fold it into its own POLYBENCH_CFLAGS.
POLYBENCH_ROOT_CFLAGS=""
POLYBENCH_LFLAGS="${POLYBENCH_LFLAGS:-}"
if [ "$(id -u)" -eq 0 ]; then
    POLYBENCH_ROOT_CFLAGS="-DPOLYBENCH_LINUX_FIFO_SCHEDULER"
    POLYBENCH_LFLAGS="$POLYBENCH_LFLAGS -lc"
fi

# --- Strict FP flags — keep ref and opt in sync ----------------------------
# Without these, FMA contraction and reordered reductions make iterative
# kernels diverge even though both binaries are IEEE-correct.
CLANG_STRICT_FP="${CLANG_STRICT_FP:--ffp-contract=off -fno-vectorize -fno-slp-vectorize}"
GCC_STRICT_FP="${GCC_STRICT_FP:--ffp-contract=off -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize}"

# Silence the noisy (harmless) warnings clang emits when it parses GCC's omp.h
# (the __malloc__ attribute signature it does not implement). Both clang and
# gcc accept -Wno-ignored-attributes.
WARN_SUPPRESS="${WARN_SUPPRESS:--Wno-ignored-attributes}"

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

# --- Kernel lists ----------------------------------------------------------
# Bundled in this repo (paths relative to $POLYBENCH).
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

# Pick the kernel list: an explicit KERNELS override wins, then SUITE selects
# the bundled set or the full suite. Result lands in the KERNEL_LIST array.
select_kernels() {
    if [ -n "${KERNELS:-}" ]; then
        read -ra KERNEL_LIST <<< "$KERNELS"
    elif [ "$SUITE" = "full" ]; then
        KERNEL_LIST=("${ALL_KERNELS[@]}")
    else
        KERNEL_LIST=("${BUNDLED_KERNELS[@]}")
    fi
}

# Resolve a kernel argument: an absolute/relative path, or a path under
# $POLYBENCH. Echoes the resolved path; non-zero if it cannot be found.
resolve_src() {
    local k="$1"
    if [ -f "$k" ]; then echo "$k"; return 0; fi
    if [ -f "$POLYBENCH/$k" ]; then echo "$POLYBENCH/$k"; return 0; fi
    return 1
}

# --- opt: the CIR/MLIR pipeline --------------------------------------------
# compile_opt <src> <outdir> <binname> <omp:on|off>
#   omp=on  -> parallel (constructs lowered through mlir-opt-omp)
#   omp=off -> sequential baseline of the *same* pipeline (no -fopenmp, so the
#              front-end never emits omp.* ops)
compile_opt() {
    local src="$1" outdir="$2" binname="$3" omp="${4:-on}"
    local name; name="$(basename "${src%.c}")"
    local tmpdir; tmpdir=$(mktemp -d)
    mkdir -p "$outdir"

    # OpenMP toggle: front-end flag, link flag and runtime libs follow it.
    local omp_cir="" link_omp="" link_libs="-lm"
    if [ "$omp" = "on" ]; then
        omp_cir="-fopenmp"; link_omp="$OPT_FOPENMP"; link_libs="$OPT_EXTRA_LIBS"
    fi

    echo "  compiling $name (opt, runtime=$RUNTIME, omp=$omp) ..."

    # 1) clang -> CIR
    "$CLANG" -S $CLANG_STRICT_FP $WARN_SUPPRESS \
        -Xclang -fclangir -Xclang -emit-cir $omp_cir \
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
    "$CLANG" -O3 $CLANG_STRICT_FP $WARN_SUPPRESS \
        -c "$INC/polybench.c" -I"$INC" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        -o "$tmpdir/polybench.o" || { rm -rf "$tmpdir"; return 1; }

    # 9) link — seq builds also pull in the serial OpenMP stubs so kernels that
    #    call omp_get_thread_num()/omp_get_num_threads() unconditionally resolve.
    local stub=""
    [ "$omp" = "off" ] && stub="$OMP_STUBS_SRC"
    "$CLANG" -O3 $CLANG_STRICT_FP \
        $link_omp -no-pie \
        "$tmpdir/$name.o" "$tmpdir/polybench.o" $stub \
        $link_libs $POLYBENCH_LFLAGS -o "$outdir/$binname" \
        || { rm -rf "$tmpdir"; return 1; }

    cp "$tmpdir/$name.opt.ll" "$outdir/$binname.ll"   # keep final IR for debugging
    rm -rf "$tmpdir"
    [ -f "$outdir/$binname" ]
}

# --- ref: stock OpenMP compiler --------------------------------------------
# compile_ref <src> <outdir> <binname> <omp:on|off>
compile_ref() {
    local src="$1" outdir="$2" binname="$3" omp="${4:-on}"
    mkdir -p "$outdir"

    local omp_flag="" stub=""
    if [ "$omp" = "on" ]; then
        omp_flag="-fopenmp"
    else
        stub="$OMP_STUBS_SRC"   # serial OpenMP stubs for the seq baseline
    fi

    echo "  compiling $(basename "${src%.c}") (ref, $REF_CC, omp=$omp) ..."

    "$REF_CC" -O3 $REF_FP $WARN_SUPPRESS $omp_flag \
        -I"$INC" -I"$(dirname "$src")" $REF_OMP_INC \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" "$INC/polybench.c" $stub \
        -lm $POLYBENCH_LFLAGS -o "$outdir/$binname" || return 1
    [ -f "$outdir/$binname" ]
}
