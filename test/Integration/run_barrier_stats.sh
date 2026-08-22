#!/bin/bash
# =============================================================================
# run_barrier_stats.sh — how many team barriers --omp-barrier-elim removes
#
# Two counts per kernel, and they are meant to agree:
#
#   what the pass claims  --mlir-pass-statistics, on the omp dialect: explicit
#                         omp.barrier ops erased, plus wsloop implicit barriers
#                         dropped by setting nowait
#   what the code shows   the runtime barrier calls left in the fully lowered
#                         IR, counted with the pass off and again with it on
#
# The first is the transform reporting on itself; the second is what survives
# to the emitted program, which is what actually costs cycles. They are not the
# same claim: an implicit barrier is dropped by setting `nowait`, and only the
# runtime's wsloop rule (`when not nowait => call ...`) turns that into one
# less call. A row where the two disagree is flagged MISMATCH and fails the
# run — either a barrier the pass claimed is emitted anyway, or the lowering
# changed something else along with it.
#
# Nothing is executed: the kernels only get as far as MLIR. Both lowerings are
# kept under $OUTDIR/<runtime>/<kernel>/barrier-stats/ so a surprising row can
# be diffed by hand — base.mlir vs elim.mlir.
#
# The claim half is runtime-independent (the pass reads neither the DSL nor a
# runtime, so the same barriers go on all three). The emitted half is not: it
# lowers for $RUNTIME and counts that runtime's barrier call, so run it once
# per runtime you care about.
#
# Either way the count is static: barriers gone from the program text, not
# executions saved. One inside a sequential outer loop counts once here and
# fires once per iteration at run time, so the dynamic saving is the larger
# number — floyd-warshall being the clearest case.
#
# Usage:
#   ./run_barrier_stats.sh                       # all kernels in the suite
#   SUITE=full ./run_barrier_stats.sh            # the external PolyBench set
#   RUNTIME=libgomp ./run_barrier_stats.sh       # count GOMP_barrier instead
#   ./run_barrier_stats.sh path/to/kernel-omp.c  # a single kernel
#   VERBOSE=1 ./run_barrier_stats.sh             # let the tools report why
#
# Exits non-zero if any kernel failed to build or came out MISMATCH, so it can
# gate CI the way run_correctness.sh does.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Nothing is built for the target or executed, so RUNTIME=pmsis wants the
# lowering rules but not the GAP SDK. Set before sourcing, like DATASET_DEFAULT.
SKIP_PULP_SDK=1
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

# Set by each driver, as native.sh expects. Nothing is ever executed here —
# the kernels only get as far as MLIR — so neither the array dumps nor the
# timer of the other drivers are wanted.
POLYBENCH_CFLAGS="$POLYBENCH_ROOT_CFLAGS"

# Split per runtime like the other drivers: the emitted half below counts one
# runtime's barrier call, so an iomp run must not overwrite a libgomp one.
OUTDIR="${OUTDIR:-$PWD/results}/$RUNTIME"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_barrier_stats.csv"

# The call each runtime's rules emit for a team barrier — both for the explicit
# `barrier` construct and in the wsloop `post` block, which is why one symbol
# per runtime is enough. Keep in step with rules.dsl.
case "$RUNTIME" in
    iomp)    BARRIER_SYM="__kmpc_barrier" ;;
    libgomp) BARRIER_SYM="GOMP_barrier" ;;
    pmsis)   BARRIER_SYM="ext_pi_cl_team_barrier" ;;
esac

# The front-end failures are noise in a batch run and the whole story when one
# kernel misbehaves.
ERRSINK="/dev/null"
[ -n "${VERBOSE:-}" ] && ERRSINK="/dev/stderr"

# The two lowerings differ by --omp-barrier-elim and nothing else: same DSL,
# same runtime, same pass order as native.sh step 3. BARRIER_ELIM from run.env
# is deliberately ignored — this script needs both configurations, so it sets
# the flag itself.
lower_kernel() {   # $1 = input mlir, $2 = output mlir, $3 = extra flag or ""
    "$MLIR_OPT_OMP" "$1" \
        --allow-unregistered-dialect \
        --omp-lower-dsl="$RULES" \
        --omp-lower-runtime="$RUNTIME" \
        ${3:+$3} \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        -o "$2" 2>"$ERRSINK"
}

# grep -c counts matching *lines*, and lowered IR can carry two calls on one;
# count occurrences instead.
count_occurrences() {   # $1 = file, $2 = fixed string
    grep -o -F -- "$2" "$1" | wc -l
}

# A single kernel may be named on the command line, as the other drivers allow.
if [ $# -gt 0 ]; then
    KERNEL_LIST=("$@")
else
    select_kernels
fi

echo "=== BARRIER ELIMINATION STATISTICS ==="
echo "runtime:   $RUNTIME (counting '$BARRIER_SYM' call sites)"
echo "polybench: $POLYBENCH"
echo "dataset:   $DATASET"
echo "kernels:   ${#KERNEL_LIST[@]}"
echo

echo "kernel,wsloops,explicit_removed,implicit_removed,calls_base,calls_elim,calls_removed,check" > "$CSV"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

FAILED=0
TOTAL_CLAIMED=0
TOTAL_REMOVED=0

for k in "${KERNEL_LIST[@]}"; do
    src="$(resolve_src "$k")" || { echo "  SKIP (not found): $k"; continue; }
    name="$(basename "$src" .c)"
    dumpdir="$OUTDIR/$name/barrier-stats"
    mkdir -p "$dumpdir"

    # Front-end, mirroring native.sh steps 1 and 2.
    "$CLANG" -S $CLANG_STRICT_FP $WARN_SUPPRESS \
        -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC" -I"$(dirname "$src")" ${INC_OMP:+-I"$INC_OMP"} \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" -o "$tmpdir/$name.cir" 2>"$ERRSINK" \
        || { echo "  ERROR (clang): $name"; echo "$name,,,,,,,ERROR" >> "$CSV"; FAILED=1; continue; }

    "$CIR_OPT" "$tmpdir/$name.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$tmpdir/$name.mlir" 2>"$ERRSINK" \
        || { echo "  ERROR (cir-opt): $name"; echo "$name,,,,,,,ERROR" >> "$CSV"; FAILED=1; continue; }
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$tmpdir/$name.mlir"

    # Total work-sharing loops, so removals read as a fraction.
    wsloops="$(count_occurrences "$tmpdir/$name.mlir" 'omp.wsloop')"

    # --- What the pass claims ------------------------------------------------
    # Statistics land on stderr, so the exit status is the only thing that
    # separates "nothing to remove" from "the pass never ran". Without the
    # check a crash reads as a clean 0.
    "$MLIR_OPT_OMP" "$tmpdir/$name.mlir" \
        --allow-unregistered-dialect --omp-barrier-elim \
        --mlir-pass-statistics -o /dev/null 2>"$tmpdir/$name.stats" \
        || { echo "  ERROR (barrier-elim): $name"; cat "$tmpdir/$name.stats" >"$ERRSINK"
             echo "$name,$wsloops,,,,,,ERROR" >> "$CSV"; FAILED=1; continue; }

    # Lines read "(S) <count> <stat-name> - <description>"; matching the name
    # as a whole field keeps a future stat that merely starts the same
    # (implicit-barriers-kept, say) from being picked up instead.
    explicit="$(awk '$3 == "explicit-barriers-removed" { print $2 }' "$tmpdir/$name.stats")"
    implicit="$(awk '$3 == "implicit-barriers-removed" { print $2 }' "$tmpdir/$name.stats")"
    explicit="${explicit:-0}"; implicit="${implicit:-0}"

    # --- What the code shows -------------------------------------------------
    lower_kernel "$tmpdir/$name.mlir" "$dumpdir/base.mlir" "" \
        || { echo "  ERROR (lowering, baseline): $name"; echo "$name,$wsloops,$explicit,$implicit,,,,ERROR" >> "$CSV"; FAILED=1; continue; }
    lower_kernel "$tmpdir/$name.mlir" "$dumpdir/elim.mlir" "--omp-barrier-elim" \
        || { echo "  ERROR (lowering, barrier-elim): $name"; echo "$name,$wsloops,$explicit,$implicit,,,,ERROR" >> "$CSV"; FAILED=1; continue; }

    # The extern declaration prints as `func.func private @<sym>(...)`, so
    # anchoring on `call @` counts call sites and not the declaration.
    calls_base="$(count_occurrences "$dumpdir/base.mlir" "call @$BARRIER_SYM(")"
    calls_elim="$(count_occurrences "$dumpdir/elim.mlir" "call @$BARRIER_SYM(")"

    claimed=$((explicit + implicit))
    removed=$((calls_base - calls_elim))
    if [ "$claimed" -eq "$removed" ]; then
        check="ok"; mark="${GREEN}ok${RESET}"
    else
        check="MISMATCH"; mark="${RED}MISMATCH${RESET}"; FAILED=1
    fi
    TOTAL_CLAIMED=$((TOTAL_CLAIMED + claimed))
    TOTAL_REMOVED=$((TOTAL_REMOVED + removed))

    printf '  %-22s %2s wsloops | claimed %s explicit + %s implicit | calls %2s -> %2s  ' \
        "$name" "$wsloops" "$explicit" "$implicit" "$calls_base" "$calls_elim"
    echo -e "$mark"
    echo "$name,$wsloops,$explicit,$implicit,$calls_base,$calls_elim,$removed,$check" >> "$CSV"
done

echo
echo "  claimed removed: $TOTAL_CLAIMED    barrier calls gone: $TOTAL_REMOVED"
echo "  Done — $CSV"

# Optional chart of the emitted half — the calls_base/calls_elim columns, which
# stand whether or not the claimed half agreed. PLOT=true, as in the other
# drivers.
render_barrier_plot "$CSV" "${CSV%.csv}.png"

if [ "$FAILED" -ne 0 ]; then
    echo -e "  ${RED}some kernels failed or came out MISMATCH${RESET} (VERBOSE=1 for the tool output)"
    exit 1
fi
