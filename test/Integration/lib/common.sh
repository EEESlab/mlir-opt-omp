#!/bin/bash
# =============================================================================
# common.sh — shared setup for the Integration test drivers
# (run_correctness.sh, run_performance.sh, tasks/run_tasks.sh).
#
# This file holds the configuration: run.env loading, tool/path resolution
# and the per-runtime knobs. The rest is sourced from sibling files at the end:
#   kernels.sh  kernel lists + selection helpers
#   native.sh   compile_opt()/compile_ref() — the host pipelines
#   pulp.sh     the PULP/gvsoc target (only when TARGET=pulp)
#
# Two execution targets, selected by RUNTIME:
#   iomp | libgomp -> TARGET=native  compile+run on the host
#   pmsis          -> TARGET=pulp    cross-compile riscv32, build+run through
#                     the PULP-SDK
# =============================================================================

# COMMON_DIR is this lib/ directory; INTEGRATION_DIR is the test/Integration
# root (run.env, the vendored kernels/ and the drivers live there);
# REPO_ROOT holds local.env and rules.dsl.
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_DIR="$(cd "$COMMON_DIR/.." && pwd)"
REPO_ROOT="$(cd "$INTEGRATION_DIR/../.." && pwd)"

# --- Colours ----------------------------------------------------------------
if [ -t 1 ]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'
    YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'
else
    GREEN=''; RED=''; CYAN=''; YELLOW=''; BOLD=''; RESET=''
fi

# --- Config ------------------------------------------------------------------
# Two files, both optional and both git-ignored:
#   <repo>/local.env      where the tools are — shared with quick-compile/
#   Integration/run.env   what to run
# The shared loader handles both, plus the tool defaults, PATH and the sanity
# checks. run.env wins over local.env, and anything already in the environment
# wins over both, so `RUNTIME=libgomp ./run_correctness.sh` still overrides them.
OMP_REPO_ROOT="$REPO_ROOT"
# shellcheck source=../../../scripts/load-local-env.sh
. "$REPO_ROOT/scripts/load-local-env.sh" "$INTEGRATION_DIR/run.env"

# --- Paths -----------------------------------------------------------------
# Defaults to the kernels vendored in this repo, so the tests are
# self-contained. Point POLYBENCH at a full PolyBench/OMP checkout (and use
# SUITE=full) to run the whole benchmark set.
POLYBENCH="${POLYBENCH:-$INTEGRATION_DIR/kernels}"
INC="${POLYBENCH_UTIL:-$POLYBENCH/utilities}"
# Serial OpenMP stubs, linked into the sequential builds (see compile_*).
OMP_STUBS_SRC="$COMMON_DIR/omp_stubs.c"

# --- Run parameters --------------------------------------------------------
# DATASET_DEFAULT lets a driver pick its own default (e.g. LARGE for perf)
# while still letting the user/run.env override via DATASET.
RUNTIME="${RUNTIME:-iomp}"           # iomp | libgomp | pmsis (pulp/gvsoc)
__DATASET_EXPLICIT="${DATASET:-}"    # remember whether the user/config chose one
DATASET="${DATASET:-${DATASET_DEFAULT:-MINI_DATASET}}"
SUITE="${SUITE:-bundled}"            # bundled | full

# PolyBench silently maps an unknown name to LARGE_DATASET — on GAP8 that dies
# in pi_l2_malloc — and setting DATASET at all disables the pmsis guard below.
case "$DATASET" in
    MINI_DATASET|SMALL_DATASET|MEDIUM_DATASET|LARGE_DATASET|EXTRALARGE_DATASET) ;;
    *)
        echo "ERROR: DATASET='$DATASET' is not a PolyBench dataset macro." >&2
        echo "       Use the full name: MINI_DATASET, SMALL_DATASET," >&2
        echo "       MEDIUM_DATASET, LARGE_DATASET or EXTRALARGE_DATASET." >&2
        echo "       Leave it unset to take the per-driver default" >&2
        echo "       (MINI_DATASET, forced on pmsis)." >&2
        exit 2
        ;;
esac

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
        TARGET="native"
        REF_CC="$CLANG"
        REF_FP="$CLANG_STRICT_FP"
        REF_OMP_INC="-I$INC_OMP"
        OPT_FOPENMP="-fopenmp"
        OPT_EXTRA_LIBS="-lm"
        ;;
    libgomp)
        TARGET="native"
        REF_CC="$GCC"
        REF_FP="$GCC_STRICT_FP"
        REF_OMP_INC=""
        OPT_FOPENMP="-fopenmp=libgomp"
        OPT_EXTRA_LIBS="-lm -lgomp"
        ;;
    pmsis|pulp)
        RUNTIME="pmsis"
        TARGET="pulp"
        # ref is built and run by the PULP-SDK harness Makefile, not by us.
        REF_CC="pulp-sdk gcc (via make)"
        REF_FP=""; REF_OMP_INC=""; OPT_FOPENMP=""; OPT_EXTRA_LIBS=""
        # GAP8 memory is tiny: unless the user/config picked a dataset, stay on
        # MINI even where a driver defaults to LARGE (run_performance.sh).
        [ -z "$__DATASET_EXPLICIT" ] && DATASET="MINI_DATASET"
        ;;
    *)
        echo "ERROR: unknown RUNTIME='$RUNTIME' (expected iomp, libgomp or pmsis)" >&2
        exit 2
        ;;
esac
unset __DATASET_EXPLICIT

# --- Load the pieces ---------------------------------------------------------
# kernels.sh  kernel lists + select_kernels/resolve_src
# native.sh   compile_opt/compile_ref (host pipelines)
# pulp.sh     everything TARGET=pulp specific (PULP_* knobs, SDK env,
#             pulp_cell and the riscv32 kernel.o pipeline)
# shellcheck source=kernels.sh
. "$COMMON_DIR/kernels.sh"
# shellcheck source=native.sh
. "$COMMON_DIR/native.sh"
if [ "$TARGET" = "pulp" ]; then
    # shellcheck source=pulp.sh
    . "$COMMON_DIR/pulp.sh"
fi
