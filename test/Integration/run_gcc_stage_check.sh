#!/bin/bash
# =============================================================================
# run_gcc_stage_check.sh — why the gcc column of run_barrier_vs_native.sh is
# counted at -O0, checked rather than asserted
#
# run_barrier_vs_native.sh counts our barriers and clang's after -O3, but gcc's
# before it. Comparing counts taken at different stages needs a reason, and this
# script is the reason in runnable form.
#
# The claim, in two halves:
#
#   1. gcc decides the elision in the front-end, so the number of barriers the
#      program *needs* does not change with -O.
#   2. What does change with -O is how many call sites those barriers are spread
#      over, and one pass accounts for all of it: jump threading. It splits the
#      path where a thread's chunk comes out empty, and the split path carries
#      its own copy of the barrier sequence. No execution gains a barrier.
#
# Both halves follow from a single measurement, which is what this checks:
#
#   with -fno-thread-jumps, the count at -O1, -O2 and -O3 equals the count
#   at -O0 — kernel by kernel, not just in total.
#
# If that holds, -O0 is not a convenient stage to count gcc at, it is the only
# one that measures the elision instead of the CFG shape. If it ever stops
# holding — a new gcc duplicating those blocks in some other pass — this script
# fails and the reason in the paper needs revisiting.
#
# Counting note: from -O2 gcc turns the last barrier of a region into a tail
# call, which prints as `jmp GOMP_barrier` and which a regex anchored on `call`
# drops. Everything here counts both. On the suite that is 6 call sites the
# `call`-only count misses at -O3, and none at -O0.
#
# Phase 1 runs a self-contained reproducer (gcc-stage/jump-threading.c) that
# needs nothing but a gcc with OpenMP. Phase 2 runs the PolyBench suite, and is
# skipped when PolyBench is not configured — so a reviewer can get the argument
# with no setup at all.
#
# Usage:
#   ./run_gcc_stage_check.sh                       # reproducer, then the suite
#   SUITE=full ./run_gcc_stage_check.sh            # the external PolyBench set
#   ./run_gcc_stage_check.sh path/to/kernel-omp.c  # a single kernel
#   GCC=gcc-13 ./run_gcc_stage_check.sh            # pin a compiler
#
# Exits non-zero if the invariant breaks, so it can gate CI.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Nothing is executed and nothing is built for the target.
SKIP_PULP_SDK=1
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

# Nothing is run, so no dumps and no timer — both sides must see the same
# defines or they would not be compiling the same program.
POLYBENCH_CFLAGS="$POLYBENCH_ROOT_CFLAGS"

OUTDIR="${OUTDIR:-$PWD/results}/gcc-stage"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_gcc_stage_check.csv"

ERRSINK="/dev/null"
[ -n "${VERBOSE:-}" ] && ERRSINK="/dev/stderr"

# The pass under suspicion. -fno-tree-dominator-opts disables jump threading
# too, but it disables enough else that the count moves for other reasons; this
# is the one knob that isolates it.
NO_JT="-fno-thread-jumps"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# Every barrier call site, tail calls included — see the header.
#
# `|| true` because grep -c exits 1 when the count is zero, and zero is the
# right answer for every kernel written with the combined directive. Without it
# they all read as compiler failures.
count_barriers() {   # $1 = .s file
    grep -cE '\b(call|jmp)\b.*GOMP_barrier' "$1" || true
}

# $1 = source, $2 = extra flags (level, plus -fno-thread-jumps or not),
# $3 = include dirs or "". Prints the count, or "-" if gcc failed.
count_at() {
    local src="$1" flags="$2" incs="$3"
    # shellcheck disable=SC2086
    "$GCC" -fopenmp $flags -S $GCC_STRICT_FP $incs \
        "$src" -o "$TMPDIR/probe.s" 2>"$ERRSINK" || { echo "-"; return 1; }
    count_barriers "$TMPDIR/probe.s"
}

FAILED=0

# --- Phase 1: the self-contained reproducer ---------------------------------
# No PolyBench, no headers. If this fails the suite half is not worth reading.

REPRO="$SCRIPT_DIR/gcc-stage/jump-threading.c"

echo "=== WHY GCC IS COUNTED AT -O0 ==="
echo "gcc:  $($GCC --version | head -1)"
echo "flags: $GCC_STRICT_FP"
echo
echo "-- reproducer: $(basename "$REPRO") (no PolyBench needed)"

if [ ! -f "$REPRO" ]; then
    echo -e "  ${RED}missing: $REPRO${RESET}" >&2
    FAILED=1
else
    r_o0="$(count_at "$REPRO" "-O0" "")"
    printf '  %-28s %s\n' "-O0" "$r_o0"
    for lvl in O1 O2 O3; do
        plain="$(count_at "$REPRO" "-$lvl" "")"
        nojt="$(count_at "$REPRO" "-$lvl $NO_JT" "")"
        if [ "$nojt" = "$r_o0" ]; then
            mark="${GREEN}ok${RESET}"
        else
            mark="${RED}BROKEN${RESET}"; FAILED=1
        fi
        printf '  %-28s %s   with %s: %s  ' "-$lvl" "$plain" "$NO_JT" "$nojt"
        echo -e "$mark"
    done
    echo "  (the middle column grows, the right one does not: the growth is"
    echo "   jump threading duplicating call sites, not barriers being added)"
fi
echo

# --- Phase 2: the suite -----------------------------------------------------
# Skipped rather than failed when PolyBench is not around: phase 1 already
# carries the argument, and a reviewer may have only a compiler.

if [ $# -gt 0 ]; then
    KERNEL_LIST=("$@")
elif [ -d "$POLYBENCH" ]; then
    select_kernels
else
    echo "-- suite: skipped (POLYBENCH not found at '$POLYBENCH')"
    echo
    KERNEL_LIST=()
fi

if [ "${#KERNEL_LIST[@]}" -gt 0 ]; then
    echo "-- suite: ${#KERNEL_LIST[@]} kernels, dataset $DATASET"
    echo "   polybench: $POLYBENCH"
    echo
    echo "kernel,o0,o1,o2,o3,o1_nojt,o2_nojt,o3_nojt,check" > "$CSV"
    printf '  %-22s %4s %4s %4s %4s | %6s %6s %6s  %s\n' \
        kernel O0 O1 O2 O3 O1njt O2njt O3njt check

    T_O0=0; T_O3=0; T_O3N=0

    for k in "${KERNEL_LIST[@]}"; do
        src="$(resolve_src "$k")" || { echo "  SKIP (not found): $k"; continue; }
        name="$(basename "$src" .c)"
        incs="-I$INC -I$(dirname "$src") -D$DATASET $POLYBENCH_CFLAGS"

        o0="$(count_at "$src" "-O0" "$incs")" \
            || { echo "  ERROR (gcc): $name"; echo "$name,,,,,,,,ERROR" >> "$CSV"; FAILED=1; continue; }
        o1="$(count_at "$src" "-O1" "$incs")"
        o2="$(count_at "$src" "-O2" "$incs")"
        o3="$(count_at "$src" "-O3" "$incs")"
        n1="$(count_at "$src" "-O1 $NO_JT" "$incs")"
        n2="$(count_at "$src" "-O2 $NO_JT" "$incs")"
        n3="$(count_at "$src" "-O3 $NO_JT" "$incs")"

        # The whole claim, per kernel: threading off puts every level back on
        # the front-end's number.
        if [ "$n1" = "$o0" ] && [ "$n2" = "$o0" ] && [ "$n3" = "$o0" ]; then
            check="ok"; mark="${GREEN}ok${RESET}"
        else
            check="BROKEN"; mark="${RED}BROKEN${RESET}"; FAILED=1
        fi

        T_O0=$((T_O0 + o0)); T_O3=$((T_O3 + o3)); T_O3N=$((T_O3N + n3))
        printf '  %-22s %4s %4s %4s %4s | %6s %6s %6s  ' \
            "$name" "$o0" "$o1" "$o2" "$o3" "$n1" "$n2" "$n3"
        echo -e "$mark"
        echo "$name,$o0,$o1,$o2,$o3,$n1,$n2,$n3,$check" >> "$CSV"
    done

    echo
    echo "  -O0: $T_O0    -O3: $T_O3    -O3 $NO_JT: $T_O3N"
    echo "  Done — $CSV"
fi

echo
if [ "$FAILED" -ne 0 ]; then
    echo -e "  ${RED}the invariant does not hold on this gcc${RESET} — the reason given for"
    echo "  counting gcc at -O0 needs revisiting before it is quoted."
    exit 1
fi
echo -e "  ${GREEN}Holds${RESET}: with $NO_JT every level agrees with -O0, kernel by kernel."
echo "  The elision is decided in the front-end; the extra call sites at -On are"
echo "  jump threading, so -O0 is the stage that measures the elision."
exit 0
