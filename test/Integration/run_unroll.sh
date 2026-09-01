#!/bin/bash
# run_unroll.sh — what the CIR unroll-by-two pass is worth on GAP8: Figure 8.
#
# Per kernel it builds the parallel binary twice, once with the CIR pass and
# once without, runs both on gvsoc and reports the cycle saving. Everything
# else is held fixed, so the delta is the pass and nothing else.
#
# Usage:
#   ./run_unroll.sh                  # the 11 kernels Figure 8 plots
#   ALL=1 ./run_unroll.sh            # the whole 30-kernel suite
#   ./run_unroll.sh path/to/kernel-omp.c
#   KERNELS="gemm-omp 2mm-omp" ./run_unroll.sh
#   CIR_UNROLL_PASS=--cir-unroll ./run_unroll.sh    # name the pass by hand
#
# RUNTIME=pmsis only. Leaves results_unroll.csv.
#
# WHAT THIS MEASURES, AND WHY THAT READING
#
# Section 4.5 calls Figure 8 a "speedup increment": 
# this driver measured the change in the parallel run time. 
# Cycles with the pass against cycles without it, on the parallel build.

#
# THE PASS
#
# The pass is not in this repository: it is a CIR pass, so it lives in the
# ClangIR fork whose cir-opt this harness calls. The driver looks for it in
# `cir-opt --help` and refuses to guess. If it is not there, there is nothing
# to measure and it says so rather than reporting a delta of zero.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${RUNTIME:-}" ] && [ "$RUNTIME" != "pmsis" ]; then
    echo "ERROR: RUNTIME=$RUNTIME — Figure 8 is a GAP8 result, see the header." >&2
    exit 2
fi
RUNTIME=pmsis
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

OUTDIR="${OUTDIR:-$PWD/results}/$RUNTIME"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_unroll.csv"
ARTDIR="$OUTDIR/unroll"
rm -rf "$ARTDIR"
mkdir -p "$ARTDIR"

REFERENCE="$SCRIPT_DIR/reference/reference.csv"

# --- Which pass ---------------------------------------------------------------

resolve_unroll_pass() {
    if [ -n "${CIR_UNROLL_PASS:-}" ]; then
        echo "$CIR_UNROLL_PASS"
        return 0
    fi
    local found
    found="$("$CIR_OPT" --help 2>&1 \
        | grep -oE -- '--cir-[a-z0-9-]*unroll[a-z0-9-]*' | sort -u)"
    case "$(echo "$found" | grep -c .)" in
        0) return 1 ;;
        1) echo "$found"; return 0 ;;
        *) echo "$found" >&2; return 2 ;;
    esac
}

UNROLL_FLAG="$(resolve_unroll_pass)"
case "$?" in
    1)
        echo "ERROR: $CIR_OPT has no CIR unrolling pass." >&2
        echo "       Figure 8 measures a CIR-level unroll-by-two pass. It is not" >&2
        echo "       part of this repository — it belongs to the ClangIR fork —" >&2
        echo "       and this cir-opt build does not carry it, so there is" >&2
        echo "       nothing here to turn on." >&2
        echo >&2
        echo "       Build a cir-opt that has it, or name it explicitly if it is" >&2
        echo "       spelled in a way the search missed:" >&2
        echo "         CIR_UNROLL_PASS=--your-pass ./run_unroll.sh" >&2
        echo >&2
        echo "       What this cir-opt does offer:" >&2
        "$CIR_OPT" --help 2>&1 | grep -oE -- '--cir-[a-z0-9-]*' | sort -u \
            | sed 's/^/         /' >&2
        exit 2
        ;;
    2)
        echo "ERROR: more than one candidate pass in $CIR_OPT (listed above)." >&2
        echo "       Name the one Figure 8 used:" >&2
        echo "         CIR_UNROLL_PASS=--the-one ./run_unroll.sh" >&2
        exit 2
        ;;
esac

# --- Which kernels ------------------------------------------------------------
# Figure 8 plots 11 of the 30: the paper says the rest are "only marginally
# affected". The figure decides the default set, read from the column it was
# extracted into, so the two cannot drift apart.
figure_kernels() {
    [ -f "$REFERENCE" ] || return 1
    awk -F';' '
        /^#/ { next }
        NR_H == 0 { for (i = 1; i <= NF; i++) if ($i == "fig8_unroll_pct") col = i
                    NR_H = 1; next }
        col && $col != "" { print $1 }' "$REFERENCE"
}

# short name (2mm) -> suite path (linear-algebra/kernels/2mm/2mm-omp.c)
path_for_kernel() {
    local want="$1" k
    for k in "${ALL_KERNELS[@]}"; do
        [ "$(basename "$k" -omp.c)" = "$want" ] && { echo "$k"; return 0; }
    done
    return 1
}

FIG8_KERNELS=()
while IFS= read -r k; do
    [ -n "$k" ] && FIG8_KERNELS+=("$k")
done < <(figure_kernels)

if [ $# -gt 0 ]; then
    KERNEL_LIST=("$@")
    SELECTION="command line"
elif [ -n "${KERNELS:-}" ]; then
    select_kernels
    SELECTION="KERNELS"
elif is_true "${ALL:-false}"; then
    KERNEL_LIST=("${ALL_KERNELS[@]}")
    SELECTION="whole suite"
elif [ "${#FIG8_KERNELS[@]}" -gt 0 ]; then
    KERNEL_LIST=()
    for k in "${FIG8_KERNELS[@]}"; do
        p="$(path_for_kernel "$k")" \
            || { echo "  SKIP (not in ALL_KERNELS): $k" >&2; continue; }
        KERNEL_LIST+=("$p")
    done
    SELECTION="the kernels Figure 8 plots"
else
    KERNEL_LIST=("${ALL_KERNELS[@]}")
    SELECTION="whole suite (no reference.csv to narrow it)"
fi

echo "=== CIR UNROLLING: WITH THE PASS vs WITHOUT (Figure 8) ==="
echo "runtime:   $RUNTIME (fixed)    dataset: $DATASET"
echo "pass:      $UNROLL_FLAG"
echo "polybench: $POLYBENCH"
echo "kernels:   ${#KERNEL_LIST[@]} — $SELECTION"
echo "files:     $ARTDIR"
echo "  One run per cell: gvsoc is deterministic, so a repetition returns the"
echo "  same cycle count."
echo

echo "kernel;base_cyc;unroll_cyc;delta_pct;published_pct" > "$CSV"
printf '  %-20s %14s %14s %9s %11s\n' kernel base unrolled 'saving %' 'Fig. 8 %'

FAILED=0

# published_pct <kernel> — what Figure 8 plots for it, empty when it plots none.
published_pct() {
    [ -f "$REFERENCE" ] || return 0
    awk -F';' -v want="$1" '
        /^#/ { next }
        NR_H == 0 { for (i = 1; i <= NF; i++) if ($i == "fig8_unroll_pct") col = i
                    NR_H = 1; next }
        col && $1 == want { print $col; exit }' "$REFERENCE"
}

for k in "${KERNEL_LIST[@]}"; do
    src="$(resolve_src "$k")" || { echo "  SKIP (not found): $k"; continue; }
    name="$(basename "$src" .c)"
    short="${name%-omp}"
    kdir="$ARTDIR/$name"
    mkdir -p "$kdir"

    CIR_UNROLL_FLAG=""
    base="$(pulp_cell "$src" opt_par "$kdir" "$kdir/base.log")" || base="NA;NA"
    CIR_UNROLL_FLAG="$UNROLL_FLAG"
    unrolled="$(pulp_cell "$src" opt_par "$kdir" "$kdir/unrolled.log")" \
        || unrolled="NA;NA"
    CIR_UNROLL_FLAG=""

    b="${base%%;*}"; u="${unrolled%%;*}"
    if [ "$b" = "NA" ] || [ "$u" = "NA" ] || [ -z "$b" ] || [ -z "$u" ]; then
        echo "  ERROR: $name"
        echo "$name;NA;NA;NA;" >> "$CSV"
        FAILED=1
        continue
    fi

    delta="$(awk -v b="$b" -v u="$u" \
        'BEGIN { printf "%.4f", 100 * (b - u) / b }')"
    pub="$(published_pct "$short")"
    printf '  %-20s %14s %14s %9s %11s\n' \
        "$short" "$b" "$u" "$delta" "${pub:--}"
    echo "$name;$b;$u;$delta;$pub" >> "$CSV"
done

# --- Against the paper --------------------------------------------------------
# The measurement, the paper's number, and the difference between them: for the
# two sentences in section 4.5 first, then for each bar of Figure 8.

summarise() {
    local n mean fw
    read -r n mean fw < <(awk -F';' '
        NR == 1 || $4 == "NA" { next }
        {
            if ($1 == "floyd-warshall-omp") { fw = $4; next }
            s += $4; n++
        }
        END { printf "%d %.4f %s", n, (n ? s / n : 0), (fw == "" ? "NA" : fw) }
    ' "$CSV")

    echo
    echo "  === SECTION 4.5 (reference/claims.csv) ==="
    printf '  %-26s %10s %10s %10s\n' quantity measured paper diff

    local rows=0 value
    show_one() {   # $1 metric  $2 subject  $3 label  $4 measured
        value=""
        read -r value _ < <(claim_row "$1" "$2")
        [ -z "${value:-}" ] && return 0
        [ "$4" = "NA" ] && return 0
        rows=$((rows + 1))
        printf '  %-26s %10s %10s %10s\n' "$3" "$4" "$value"  \
            "$(awk -v m="$4" -v p="$value" 'BEGIN { printf "%.4f", m - p }')"
    }
    show_one unroll_pct ten-apps       "mean, excl. floyd-w. %" "$mean"
    show_one unroll_pct floyd-warshall "floyd-warshall %"       "$fw"
    unset -f show_one

    if [ "$rows" -eq 0 ]; then
        echo "  no unroll_pct rows in $CLAIMS"
    else
        echo
        echo "  The mean covers the $n kernels other than floyd-warshall, which"
        echo "  the paper gives a number of its own."
    fi

    # Per kernel against the figure itself: the published values are exact, and
    # were taken on a simulator anyone can rerun, so the comparison is worth
    # making kernel by kernel rather than in aggregate.
    local pairs
    pairs="$(awk -F';' 'NR > 1 && $4 != "NA" && $5 != "" { print }' "$CSV")"
    [ -z "$pairs" ] && return 0
    echo
    echo "  === FIGURE 8 (reference/reference.csv) ==="
    printf '  %-20s %11s %11s %11s\n' kernel here published diff
    echo "$pairs" | awk -F';' '
        { printf "  %-20s %11.2f %11.2f %11.2f\n",
                 substr($1, 1, length($1) - 4), $4, $5, $4 - $5 }'
}

echo
if [ "$FAILED" -eq 0 ]; then
    summarise
else
    echo "  Not compared with the paper: a kernel failed, so the mean would be"
    echo "  over a different set than the one Figure 8 plots."
fi

echo
echo "  The LLVM IR of both builds is kept, one directory per kernel under"
echo "    $ARTDIR/<kernel>/"
echo "      <kernel>_omp-on.ll           without the pass"
echo "      <kernel>_omp-on_unrolled.ll  with it"
echo "  so what the pass did is a diff away:"
echo "    diff <(grep -c . *_omp-on.ll) <(grep -c . *_omp-on_unrolled.ll)"
echo "  Done — $CSV"

[ "$FAILED" -ne 0 ] && {
    echo "  some kernels failed: the failing tool wrote its message into the"
    echo "  cell's log above. PULP_KEEP_TMP=1 keeps the intermediate .cir/.mlir"
    echo "  so the step can be rerun by hand; PULP_VERBOSE=1 streams make/gvsoc."
    exit 1
}
exit 0
