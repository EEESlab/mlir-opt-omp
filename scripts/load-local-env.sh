#!/bin/bash
# =============================================================================
# load-local-env.sh — resolve the per-machine tool paths.
#
# Sourced by everything that drives the compiler pipeline (the test/Integration
# drivers) so they all find the tools the same way. Not meant to be executed on
# its own.
#
#   OMP_REPO_ROOT=/path/to/repo
#   . "$OMP_REPO_ROOT/scripts/load-local-env.sh" [extra-config-file ...]
#
# Loads <repo>/local.env, then any extra config files passed as arguments,
# in that order. Anything already set in the environment wins over all of them,
# so `RUNTIME=libgomp ./run_correctness.sh` still overrides the files.
#
# Callers may set OMP_DEFAULT_TOOL_BIN before sourcing: it applies only when
# neither the environment nor the config files provide OMP_TOOL_BIN.
#
# On return: LLVM_BIN/OMP_TOOL_BIN are on PATH, every tool variable holds a
# usable command, and anything obviously wrong has been reported on stderr.
# =============================================================================

if [ -z "${OMP_REPO_ROOT:-}" ]; then
    echo "load-local-env.sh: OMP_REPO_ROOT must be set before sourcing" >&2
    return 1 2>/dev/null || exit 1
fi

# Every variable the config files may set. The snapshot below walks this list to
# give the environment priority over the files, so a new knob only needs adding
# here to become overridable inline.
OMP_CONFIG_VARS="LLVM_BIN OMP_TOOL_BIN CLANG GCC OPT LLC CIR_OPT MLIR_OPT
MLIR_TRANSLATE MLIR_OPT_OMP POLYBENCH POLYBENCH_UTIL RULES INC_OMP OUTDIR
RUNTIME DATASET THREADS KERNELS BARRIER_ELIM REPS VARIANCE_ACCEPTED
PLOT PLOT_PYTHON COMPARE
POLYBENCH_LFLAGS CLANG_STRICT_FP GCC_STRICT_FP OMP_PLACES OMP_PROC_BIND
OMP_WAIT_POLICY KMP_BLOCKTIME GOMP_SPINCOUNT
WARN_SUPPRESS PULP_APP_DIR PULP_PLATFORM PULP_MAKE PULP_MAKE_ARGS PULP_OPT
PULP_LLC PULP_OPT_FLAGS PULP_LLC_FLAGS PULP_BUILD_BIN PULP_POLYBENCH_DEFS
PULP_SDK_ENV PULP_TOOLCHAIN_BIN PULP_VERBOSE"

# Self-contained so the sourcing scripts do not have to define colours.
# An empty label prints an indented continuation line.
__omp_msg() {   # $1 = label, rest = text
    local label="$1"; shift
    if [ -z "$label" ]; then
        printf '         %s\n' "$*" >&2
    elif [ -t 2 ]; then
        printf '\033[1;33m%s:\033[0m %s\n' "$label" "$*" >&2
    else
        printf '%s: %s\n' "$label" "$*" >&2
    fi
}

# --- Load the config files ---------------------------------------------------
__omp_preset=""
for __omp_v in $OMP_CONFIG_VARS; do
    if [ -n "${!__omp_v+x}" ]; then
        __omp_preset="$__omp_preset $__omp_v=$(printf '%q' "${!__omp_v}")"
    fi
done

__omp_seen=0
for __omp_f in "$OMP_REPO_ROOT/local.env" "$@"; do
    if [ -f "$__omp_f" ]; then
        if [ "$__omp_f" = "$OMP_REPO_ROOT/local.env" ]; then __omp_seen=1; fi
        # shellcheck disable=SC1090
        set -a; . "$__omp_f"; set +a
    fi
done
if [ -n "$__omp_preset" ]; then
    eval "$__omp_preset"        # inline/shell overrides reclaim priority
fi

# --- Tool locations ----------------------------------------------------------
# LLVM_BIN / OMP_TOOL_BIN, if set, go on PATH so the bare tool names below
# resolve to the right build.
LLVM_BIN="${LLVM_BIN:-}"
OMP_TOOL_BIN="${OMP_TOOL_BIN:-${OMP_DEFAULT_TOOL_BIN:-}}"
if [ -n "$LLVM_BIN" ];     then PATH="$LLVM_BIN:$PATH"; fi
if [ -n "$OMP_TOOL_BIN" ]; then PATH="$OMP_TOOL_BIN:$PATH"; fi
export PATH

CLANG="${CLANG:-clang}"
GCC="${GCC:-gcc}"
OPT="${OPT:-opt}"
LLC="${LLC:-llc}"
CIR_OPT="${CIR_OPT:-cir-opt}"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
MLIR_OPT_OMP="${MLIR_OPT_OMP:-mlir-opt-omp}"

RULES="${RULES:-$OMP_REPO_ROOT/rules.dsl}"
# OpenMP headers used by the clang->CIR front-end; must match the local GCC.
INC_OMP="${INC_OMP:-/usr/lib/gcc/x86_64-linux-gnu/13/include}"

# --- Report what is obviously wrong ------------------------------------------
# All of this fails eventually anyway, just much later and with worse messages.
__omp_check_bin() {   # $1 = variable name, $2 = directory, $3 = expected tool
    if [ -z "$2" ]; then return 0; fi
    case "$3" in /*) return 0 ;; esac   # tool set to an explicit path, not our business
    if [ ! -d "$2" ]; then
        __omp_msg WARNING "$1='$2' is not a directory — using $3 from PATH"
    elif [ ! -x "$2/$3" ]; then
        __omp_msg WARNING "$1='$2' holds no $3 — using $3 from PATH"
    fi
}
__omp_check_bin LLVM_BIN "$LLVM_BIN" "$CLANG"
__omp_check_bin OMP_TOOL_BIN "$OMP_TOOL_BIN" "$MLIR_OPT_OMP"

# A wrong GCC version here fails inside the front-end with unhelpful header
# errors, so name the directories that would actually work.
if [ ! -f "$INC_OMP/omp.h" ]; then
    __omp_msg WARNING "INC_OMP='$INC_OMP' has no omp.h — the clang->CIR step will fail"
    for __omp_d in /usr/lib/gcc/*/*/include; do
        if [ -f "$__omp_d/omp.h" ]; then __omp_msg "" "try INC_OMP=\"$__omp_d\""; fi
    done
fi

if [ "$__omp_seen" -eq 0 ]; then
    __omp_msg NOTE "no local.env in $OMP_REPO_ROOT — using the tools on PATH"
    __omp_msg "" "cp local.env.example local.env   to pin them"
fi

unset -f __omp_msg __omp_check_bin
unset __omp_preset __omp_v __omp_f __omp_d __omp_seen
