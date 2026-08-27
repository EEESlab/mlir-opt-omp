#!/bin/bash
# =============================================================================
# run_pulp_probe.sh — build and run a PULP probe on gvsoc
#
# A probe is a plain C file under pulp-probes/ that provides cluster_main() and
# answers a question about the runtime rather than about a kernel. It goes
# through the harness the same way an ordinary kernel does, so this driver
# builds and runs it exactly as the other drivers do — no mlir-opt-omp in the
# path, no kernel.o: the probe *is* the kernel.
#
# It exists because doing it by hand means remembering the SDK env, the RISC-V
# toolchain on PATH, and `platform=`. common.sh and lib/pulp.sh already know all
# three, and they are the same source of truth run_correctness.sh uses.
#
# CELL picks how the probe is built, and which question it therefore answers:
#
#   ref_seq  (default)  SDK gcc, no OpenMP — for a probe written against the
#                       ext_pi_* shims directly, like barrier-probe.c
#   ref_par             SDK gcc with its own OpenMP — the control
#   opt_par             ClangIR -> mlir-opt-omp -> PULP_LLC, our lowering
#
# A probe written with real pragmas run under both ref_par and opt_par is what
# separates "this shape is broken" from "we lower this shape wrongly".
#
# Usage:
#   ./run_pulp_probe.sh                       # pulp-probes/barrier-probe.c
#   ./run_pulp_probe.sh pulp-probes/other.c   # any probe
#   CELL=opt_par ./run_pulp_probe.sh pulp-probes/omp-shape-probe.c
#   PULP_VERBOSE=1 ./run_pulp_probe.sh        # stream the build/run output
#
# The console log is kept whole: a probe that hangs is diagnosed by the last
# line it printed, so nothing here filters it.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The probe runs on the cluster, so the SDK and the toolchain are wanted.
RUNTIME="${RUNTIME:-pmsis}"
# Stream by default, the opposite of the other drivers. A probe is watched, not
# summarised: the thing it is asked about is where it stops, and a probe that
# stops never reaches the code that would print the log at the end. Set
# PULP_VERBOSE=0 to send it all to the log instead.
PULP_VERBOSE="${PULP_VERBOSE:-1}"
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

if [ "$TARGET" != "pulp" ]; then
    echo "ERROR: a probe runs on the cluster — use RUNTIME=pmsis." >&2
    exit 2
fi

PROBE="${1:-$SCRIPT_DIR/pulp-probes/barrier-probe.c}"
[ -f "$PROBE" ] || { echo "ERROR: no such probe: $PROBE" >&2; exit 2; }
PROBE="$(cd "$(dirname "$PROBE")" && pwd)/$(basename "$PROBE")"
NAME="$(basename "$PROBE" .c)"

CELL="${CELL:-ref_seq}"
case "$CELL" in
    ref_seq|ref_par|opt_seq|opt_par) ;;
    *) echo "ERROR: CELL='$CELL' — use ref_seq, ref_par, opt_seq or opt_par." >&2
       exit 2 ;;
esac

# One directory per cell: the same probe under ref_par and opt_par is the whole
# point, and a second run must not overwrite the log that answered the first.
OUTDIR="${OUTDIR:-$PWD/results}/pulp-probes/$NAME/$CELL"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$NAME.log"
: > "$LOG"

echo "=== PULP PROBE: $NAME ==="
echo "probe:     $PROBE"
echo "cell:      $CELL"
echo "app dir:   $PULP_APP_DIR"
echo "platform:  $PULP_PLATFORM"
echo "log:       $LOG"
echo

# The cycle count pulp_cell returns is meaningless for a probe and is dropped;
# what matters is the console output and whether the run came back.
if pulp_cell "$PROBE" "$CELL" "$OUTDIR" "$LOG" > /dev/null; then
    status=0
else
    status=1
fi

# The probe's own output is the result, whether or not the run came back clean:
# a hang leaves a truncated log, and where it stops is the answer.
echo "── probe output ──"
cat "$LOG"
echo "──────────────────"
echo

if [ "$status" -ne 0 ]; then
    echo -e "  ${YELLOW}the run did not finish cleanly${RESET} — the last line above is"
    echo "  where it stopped. Full log: $LOG"
    exit 1
fi
echo "  Done — $LOG"
