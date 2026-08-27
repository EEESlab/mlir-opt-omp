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
# Usage:
#   ./run_pulp_probe.sh                       # pulp-probes/barrier-probe.c
#   ./run_pulp_probe.sh pulp-probes/other.c   # any probe
#   PULP_VERBOSE=1 ./run_pulp_probe.sh        # stream the build/run output
#
# The console log is kept whole: a probe that hangs is diagnosed by the last
# line it printed, so nothing here filters it.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The probe runs on the cluster, so the SDK and the toolchain are wanted.
RUNTIME="${RUNTIME:-pmsis}"
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

OUTDIR="${OUTDIR:-$PWD/results}/pulp-probes/$NAME"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$NAME.log"
: > "$LOG"

echo "=== PULP PROBE: $NAME ==="
echo "probe:     $PROBE"
echo "app dir:   $PULP_APP_DIR"
echo "platform:  $PULP_PLATFORM"
echo "log:       $LOG"
echo

# ref_seq is the cell that builds the kernel source straight into the app with
# neither OMP_NATIVE nor OMP_OPT — which is what a probe wants. The cycle count
# it returns is meaningless here and is dropped.
if pulp_cell "$PROBE" ref_seq "$OUTDIR" "$LOG" > /dev/null; then
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
