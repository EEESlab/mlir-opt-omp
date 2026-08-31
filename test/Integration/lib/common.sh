#!/bin/bash
# =============================================================================
# common.sh — shared setup for the Integration test drivers
# (run_correctness.sh, run_performance.sh, run_barrier_vs_native.sh,
# tasks/run_tasks.sh).
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
# root (run.env, configs/ and the drivers live there);
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
#   <repo>/local.env      where the tools are
#   Integration/run.env   what to run
# The shared loader handles both, plus the tool defaults, PATH and the sanity
# checks. run.env wins over local.env, and anything already in the environment
# wins over both, so `RUNTIME=libgomp ./run_correctness.sh` still overrides them.
#
# RUN_ENV names a config file to read *instead of* run.env — the named ones in
# configs/ reproduce a figure of the paper. Instead of, not as well as: a stale
# personal run.env silently contaminating a reproduction run is exactly the
# failure this is meant to prevent, and the command line then says which
# configuration produced the numbers rather than leaving it to whichever file
# happened to be on disk.
#
# A relative RUN_ENV is resolved against this directory as well as the cwd, so
# `RUN_ENV=configs/paper-iomp.env` works from either. A name that matches
# nothing is a hard error: the loader skips files it cannot find, which for
# run.env is right (it is optional) and for RUN_ENV would be the worst
# outcome — a reproduction run silently taken under the defaults.
OMP_RUN_ENV_FILE="$INTEGRATION_DIR/run.env"
if [ -n "${RUN_ENV:-}" ]; then
    if [ -f "$RUN_ENV" ]; then
        OMP_RUN_ENV_FILE="$RUN_ENV"
    elif [ -f "$INTEGRATION_DIR/$RUN_ENV" ]; then
        OMP_RUN_ENV_FILE="$INTEGRATION_DIR/$RUN_ENV"
    else
        echo "ERROR: RUN_ENV='$RUN_ENV' names no readable file." >&2
        echo "       Tried: $RUN_ENV" >&2
        echo "          and: $INTEGRATION_DIR/$RUN_ENV" >&2
        echo "       Available: $(cd "$INTEGRATION_DIR" 2>/dev/null \
            && echo configs/*.env)" >&2
        exit 2
    fi
fi

OMP_REPO_ROOT="$REPO_ROOT"
# shellcheck source=../../../scripts/load-local-env.sh
. "$REPO_ROOT/scripts/load-local-env.sh" "$OMP_RUN_ENV_FILE"

# --- Paths -----------------------------------------------------------------
# POLYBENCH must name a PolyBench/OMP checkout: no kernels are vendored here.
# Checked rather than assumed, because the failure otherwise is every kernel
# being skipped as "not found" — a run that reports nothing wrong and measures
# nothing at all.
POLYBENCH="${POLYBENCH:-}"
if [ -z "$POLYBENCH" ] || [ ! -d "$POLYBENCH" ]; then
    echo "ERROR: POLYBENCH must point at a PolyBench/OMP checkout." >&2
    if [ -n "$POLYBENCH" ]; then
        echo "       POLYBENCH='$POLYBENCH' is not a directory." >&2
    else
        echo "       It is not set. Put it in <repo>/local.env:" >&2
        echo "         POLYBENCH=\"/path/to/PolyBenchC-4.2.1-OpenMP\"" >&2
    fi
    exit 2
fi
INC="${POLYBENCH_UTIL:-$POLYBENCH/utilities}"
# Serial OpenMP stubs, linked into the sequential builds (see compile_*).
OMP_STUBS_SRC="$COMMON_DIR/omp_stubs.c"

# --- Run parameters --------------------------------------------------------
# DATASET_DEFAULT lets a driver pick its own default (e.g. LARGE for perf)
# while still letting the user/run.env override via DATASET.
RUNTIME="${RUNTIME:-iomp}"           # iomp | libgomp | pmsis (pulp/gvsoc)
# Redundant team-barrier elimination (--omp-barrier-elim). Off by default so a
# plain run is the baseline to compare against.
BARRIER_ELIM="${BARRIER_ELIM:-0}"    # 0 | 1 | both
case "$BARRIER_ELIM" in
    0|1|both) ;;
    *)
        echo "ERROR: BARRIER_ELIM='$BARRIER_ELIM' is not 0, 1 or both." >&2
        exit 1
        ;;
esac
# FLAG is spliced into the mlir-opt-omp command line by native.sh and pulp.sh.
# DIR_TAG is appended to the drivers' output directory, so an optimised run
# lands next to its baseline instead of on top of it. FILE_TAG does the same
# for the result files of run_performance.sh: those get copied out of that
# directory — into a paper, a slide deck — where the two configurations would
# otherwise be two files with one name. All three empty when the pass is off,
# so the baseline command line and paths stay exactly as they were.
#
# `both` is a mode rather than a setting, and only run_performance.sh knows
# what to do with it: build the kernel both ways and time the two against each
# other inside one run. FLAG stays empty here — that driver needs one build of
# each, so it sets the flag around each compile itself.
BARRIER_ELIM_FLAG=""
BARRIER_ELIM_DIR_TAG=""
BARRIER_ELIM_FILE_TAG=""
case "$BARRIER_ELIM" in
    1)
        BARRIER_ELIM_FLAG="--omp-barrier-elim"
        BARRIER_ELIM_DIR_TAG="-barrier-elim"
        BARRIER_ELIM_FILE_TAG="_barrier-elim"
        ;;
    both)
        BARRIER_ELIM_DIR_TAG="-barrier-ab"
        BARRIER_ELIM_FILE_TAG="_barrier-ab"
        ;;
esac
__DATASET_EXPLICIT="${DATASET:-}"    # remember whether the user/config chose one
DATASET="${DATASET:-${DATASET_DEFAULT:-MINI_DATASET}}"

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

#export OMP_PLACES="${OMP_PLACES:-cores}"
#export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"

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

# --- Plotting ----------------------------------------------------------------
# is_true <value> -> 0 if it reads as a boolean "yes" (1/true/yes/on), else 1.
is_true() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# Print the python a chart should be rendered with, or return non-zero after
# saying why. Order: $PLOT_PYTHON, the local venv, then python3 on PATH.
# Every caller treats the failure as a skipped plot rather than a failed run —
# the CSV is the result, the figure is a convenience.
plot_python() {
    local py
    if [ -n "${PLOT_PYTHON:-}" ]; then
        py="$PLOT_PYTHON"
    elif [ -x "$INTEGRATION_DIR/.venv/bin/python" ]; then
        py="$INTEGRATION_DIR/.venv/bin/python"
    else
        py="$(command -v python3 || command -v python)" || {
            echo -e "${YELLOW}[plot] python3 not found — skipping plot${RESET}" >&2
            return 1
        }
    fi
    if ! "$py" -c 'import matplotlib' >/dev/null 2>&1; then
        echo -e "${YELLOW}[plot] matplotlib not available in $py — skipping plot.${RESET}" >&2
        echo -e "${YELLOW}[plot] set it up once with:${RESET}" >&2
        echo -e "${YELLOW}[plot]   python3 -m venv $INTEGRATION_DIR/.venv${RESET}" >&2
        echo -e "${YELLOW}[plot]   $INTEGRATION_DIR/.venv/bin/pip install matplotlib numpy${RESET}" >&2
        return 1
    fi
    printf '%s' "$py"
}

# --- Load the pieces ---------------------------------------------------------
# kernels.sh  kernel lists + select_kernels/resolve_src
# native.sh   compile_opt/compile_ref (host pipelines)
# pulp.sh     everything TARGET=pulp specific (PULP_* knobs, SDK env,
#             pulp_cell and the riscv32 kernel.o pipeline)
# shellcheck source=kernels.sh
. "$COMMON_DIR/kernels.sh"
# shellcheck source=native.sh
. "$COMMON_DIR/native.sh"
# SKIP_PULP_SDK is for a driver that only goes as far as MLIR
# (run_barrier_vs_native.sh): RUNTIME=pmsis then needs the lowering rules but
# neither the GAP SDK nor PULP_APP_DIR, which pulp.sh insists on.
if [ "$TARGET" = "pulp" ] && [ -z "${SKIP_PULP_SDK:-}" ]; then
    # shellcheck source=pulp.sh
    . "$COMMON_DIR/pulp.sh"
fi
