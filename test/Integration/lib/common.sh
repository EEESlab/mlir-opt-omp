#!/bin/bash
# common.sh — shared setup for the Integration drivers. Sourced, not run.
#
# Loads the config (local.env, then run.env or $RUN_ENV), resolves the tools and
# the per-runtime knobs, then pulls in kernels.sh, native.sh and — on pmsis —
# pulp.sh. RUNTIME picks the target: iomp/libgomp run on the host, pmsis
# cross-compiles for PULP/GAP8 and runs on gvsoc.
#
# Every variable and its default: README.md, "Configuration reference".

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

POLYBENCH_ROOT_CFLAGS=""
POLYBENCH_LFLAGS="${POLYBENCH_LFLAGS:-}"
if [ "$(id -u)" -eq 0 ]; then
    POLYBENCH_ROOT_CFLAGS="-DPOLYBENCH_LINUX_FIFO_SCHEDULER"
    POLYBENCH_LFLAGS="$POLYBENCH_LFLAGS -lc"
fi

CLANG_STRICT_FP="${CLANG_STRICT_FP:--ffp-contract=off -fno-vectorize -fno-slp-vectorize}"
GCC_STRICT_FP="${GCC_STRICT_FP:--ffp-contract=off -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize}"

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

# --- The paper's own claims -------------------------------------------------
# reference/claims.csv holds what the paper states in prose, one row per
# sentence, so a driver can check its own totals against it instead of leaving
# them to be read off the screen. reference/reference.csv is the other half:
# what the figures plot. See reference/README.md.
CLAIMS="$INTEGRATION_DIR/reference/claims.csv"

# claim_row <metric> <subject> -> "value tolerance", empty when the file or the
# row is missing. The header line cannot collide: its 4th field is the literal
# "metric", which is not one.
claim_row() {
    [ -f "$CLAIMS" ] || return 0
    awk -F';' -v m="$1" -v s="$2" \
        '!/^#/ && $4 == m && $5 == s { print $7, $8; exit }' "$CLAIMS"
}

# There is deliberately no claim_verdict: the drivers print the measured value
# next to the paper's and stop there. Whether a difference matters is a
# judgement about the work, and it belongs to whoever is reading.

. "$COMMON_DIR/kernels.sh"
# shellcheck source=native.sh
. "$COMMON_DIR/native.sh"
# SKIP_PULP_SDK is for a driver that only goes as far as MLIR
if [ "$TARGET" = "pulp" ] && [ -z "${SKIP_PULP_SDK:-}" ]; then
    # shellcheck source=pulp.sh
    . "$COMMON_DIR/pulp.sh"
fi
