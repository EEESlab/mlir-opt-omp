#!/bin/bash
# run_performance.sh — builds a 2x2 matrix per kernel (sequential/parallel x
# native/ours), times each cell with PolyBench's cycle-accurate timer, and
# reports the self-relative parallel speedup of each toolchain.
#
# Usage:
#   ./run_performance.sh                            # all kernels, defaults
#   RUNTIME=libgomp ./run_performance.sh            # iomp | libgomp | pmsis
#   RUN_ENV=configs/paper-iomp.env ./run_performance.sh   # reproduce a figure
#   ./run_performance.sh path/to/kernel-omp.c
#   REPS=5 ./run_performance.sh                     # fewer timed runs, faster
#   PLOT=true ./run_performance.sh                  # + the chart(s)
#   BARRIER_ELIM=1|both ./run_performance.sh        # with the pass, or A/B
#   COMPARE=false ./run_performance.sh              # skip the paper comparison
#   KEEP=1 ./run_performance.sh                     # keep binaries and logs
#
# Leaves results_performance.csv: one row per kernel plus a GEOMEAN row. The
# cycle counts are the result; the binaries and per-repetition logs behind them
# go unless KEEP=1. Configuration: README.md, "Configuration reference".

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Perf wants a real workload by default; MINI is dominated by thread-spawn cost.
# DATASET / run.env still override this (see common.sh).
DATASET_DEFAULT="LARGE_DATASET"
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

# --- Performance-specific config -------------------------------------------
THREADS="${THREADS:-16}"             # thread count for the parallel cells
REPS="${REPS:-10}"                    # timed runs per cell (>=3; min+max dropped)
VARIANCE_ACCEPTED="${VARIANCE_ACCEPTED:-5}"   # rel std-dev warn threshold (%)
OUTDIR="${OUTDIR:-$PWD/results}/$RUNTIME$BARRIER_ELIM_DIR_TAG"
PLOT="${PLOT:-false}"                # true -> render a speedup bar chart at the end

# Performance times the kernel with the cycle-accurate TSC timer. This is
# mutually exclusive with -DPOLYBENCH_DUMP_ARRAYS, hence its own CFLAGS.
POLYBENCH_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_CYCLE_ACCURATE_TIMER $POLYBENCH_ROOT_CFLAGS"

bench_stats() {
    local logfile="$1"
    local keep=$((REPS - 2))
    [ "$keep" -lt 1 ] && keep=$REPS   # REPS<3: keep them all
    sort -g "$logfile" | head -n $((REPS == keep ? REPS : REPS - 1)) \
        | tail -n "$keep" | awk '
        /^[0-9.eE+-]+$/ { v[++n] = $1; s += $1; if (n == 1 || $1 < mn) mn = $1 }
        END {
            if (n < 1) exit 1
            m = s / n
            for (i = 1; i <= n; i++) { d = v[i] - m; ss += d * d }
            sd = sqrt(ss / n)
            rsd = (m != 0) ? sd / m * 100 : 0
            printf "%.6f %.6f %.6f %.4f", m, sd, mn, rsd
        }'
}

# bench_noisy <label> <relsd> — warn when a cell's spread says it is measuring
# the machine rather than the code.
bench_noisy() {
    local warn
    warn=$(awk -v r="$2" -v t="$VARIANCE_ACCEPTED" 'BEGIN { print (r < t) ? "ok" : "warn" }')
    [ "$warn" = "warn" ] && \
        echo -e "    ${YELLOW}[noisy]${RESET} $1: rel std-dev $2% > ${VARIANCE_ACCEPTED}%" >&2
    return 0
}

run_benchmark() {
    local binary="$1" nthreads="$2" label="$3" logfile="$4"
    BENCH_STDDEV="NA"; BENCH_MIN="NA"; BENCH_RELSD="NA"

    echo -e "    ${CYAN}timing ${label} (${nthreads}T) x${REPS}...${RESET}" >&2
    : > "$logfile"
    local i
    for ((i = 0; i < REPS; i++)); do
        OMP_NUM_THREADS="$nthreads" "$binary" >> "$logfile" 2>/dev/null \
            || { echo -e "    ${RED}run failed${RESET}" >&2; return 1; }
    done

    local stats; stats="$(bench_stats "$logfile")"
    if [ -z "$stats" ]; then
        echo -e "    ${RED}no numeric output — is -DPOLYBENCH_TIME set?${RESET}" >&2
        return 1
    fi

    local mean
    read -r mean BENCH_STDDEV BENCH_MIN BENCH_RELSD <<< "$stats"
    bench_noisy "$label" "$BENCH_RELSD"
    echo -e "    ${BOLD}${label}: ${mean} cyc${RESET} (sd ${BENCH_STDDEV}, min ${BENCH_MIN})" >&2

    printf "%s" "$mean"
}

# run_benchmark_ab <binA> <binB> <nthreads> <logA> <logB>
run_benchmark_ab() {
    local a="$1" b="$2" nthreads="$3" loga="$4" logb="$5"

    echo -e "    ${CYAN}timing base/elim, alternati (${nthreads}T) x${REPS}...${RESET}" >&2
    : > "$loga"; : > "$logb"
    local i
    for ((i = 0; i < REPS; i++)); do
        OMP_NUM_THREADS="$nthreads" "$a" >> "$loga" 2>/dev/null \
            || { echo -e "    ${RED}run failed (base)${RESET}" >&2; return 1; }
        OMP_NUM_THREADS="$nthreads" "$b" >> "$logb" 2>/dev/null \
            || { echo -e "    ${RED}run failed (elim)${RESET}" >&2; return 1; }
    done

    local sa sb; sa="$(bench_stats "$loga")"; sb="$(bench_stats "$logb")"
    if [ -z "$sa" ] || [ -z "$sb" ]; then
        echo -e "    ${RED}no numeric output — is -DPOLYBENCH_TIME set?${RESET}" >&2
        return 1
    fi

    local ma mb sda sdb rsda rsdb skip
    read -r ma sda skip rsda <<< "$sa"
    read -r mb sdb skip rsdb <<< "$sb"
    bench_noisy base "$rsda"
    bench_noisy elim "$rsdb"
    echo -e "    ${BOLD}base: ${ma} cyc${RESET} (sd ${sda})   ${BOLD}elim: ${mb} cyc${RESET} (sd ${sdb})" >&2

    printf "%s %s %s %s" "$ma" "$sda" "$mb" "$sdb"
}

# ratio <a> <b> -> a/b with 4 decimals, or NA.
ratio() {
    awk -v a="$1" -v b="$2" \
        'BEGIN { if (b+0 == 0) print "NA"; else printf "%.4f", a / b }'
}

run_kernel_pulp() {
    local src="$1" name="$2"
    local d="$OUTDIR/$name/performance"
    mkdir -p "$d"

    echo -e "${BOLD}── $name${RESET}" >&2

    local -A CYC SIZ
    local cell res i=1
    for cell in ref_seq ref_par opt_seq opt_par; do
        echo "  [$i/4] $cell (gvsoc build+run)..." >&2
        : > "$d/$cell.log"
        if ! res="$(pulp_cell "$src" "$cell" "$d" "$d/$cell.log")"; then
            echo "  ERROR: $cell failed" >&2; emit_na "$name"; return
        fi
        CYC[$cell]="${res%%;*}"; SIZ[$cell]="${res##*;}"
        [ "${CYC[$cell]}" = "NA" ] && \
            echo -e "    ${YELLOW}WARNING: no 'Cycles =' line in $d/$cell.log${RESET}" >&2
        echo -e "    ${BOLD}$cell: ${CYC[$cell]} cyc${RESET} (elf ${SIZ[$cell]} B)" >&2
        i=$((i + 1))
    done

    # --- derived metrics (NA-propagating) -----------------------------------
    local SP_NATIVE=NA SP_OPT=NA OPT_VS_NAT_PAR=NA OPT_VS_NAT_SEQ=NA
    if [ "${CYC[ref_seq]}" != "NA" ] && [ "${CYC[ref_par]}" != "NA" ] &&
       [ "${CYC[opt_seq]}" != "NA" ] && [ "${CYC[opt_par]}" != "NA" ]; then
        SP_NATIVE=$(ratio "${CYC[ref_seq]}" "${CYC[ref_par]}")
        SP_OPT=$(ratio    "${CYC[opt_seq]}" "${CYC[opt_par]}")
        OPT_VS_NAT_PAR=$(ratio "${CYC[ref_par]}" "${CYC[opt_par]}")
        OPT_VS_NAT_SEQ=$(ratio "${CYC[ref_seq]}" "${CYC[opt_seq]}")
    fi

    {
        echo ""
        printf "  %-22s %18s %18s\n" "" "native (ref)" "our tool (opt)"
        printf "  %-22s %18s %18s\n" "seq cycles" "${CYC[ref_seq]}" "${CYC[opt_seq]}"
        printf "  %-22s %18s %18s\n" "par cycles" "${CYC[ref_par]}" "${CYC[opt_par]}"
        printf "  %-22s %18s %18s\n" "self speedup seq→par" "${SP_NATIVE}x" "${SP_OPT}x"
        printf "  %-22s %18s\n" "opt vs native (par)" "${OPT_VS_NAT_PAR}x"
        printf "  %-22s %18s\n" "opt vs native (seq)" "${OPT_VS_NAT_SEQ}x"
        echo ""
    } >&2

    echo "${name};${CYC[ref_seq]};${CYC[ref_par]};${CYC[opt_seq]};${CYC[opt_par]};${SP_NATIVE};${SP_OPT};${OPT_VS_NAT_PAR};${OPT_VS_NAT_SEQ};${SIZ[ref_seq]};${SIZ[ref_par]};${SIZ[opt_seq]};${SIZ[opt_par]}" >> "$CSV"
    prune_kernel "$d"
}

run_kernel_ab() {
    local src="$1" name="$2"
    local d="$OUTDIR/$name/performance"
    mkdir -p "$d"

    echo -e "${BOLD}── $name${RESET}" >&2

    echo "  [1/2] base (senza la pass)..." >&2
    BARRIER_ELIM_FLAG=""
    compile_opt "$src" "$d" "${name}_ab_base" on >&2 \
        || { echo "  ERROR base compile" >&2; emit_na "$name"; return; }
    echo "  [2/2] elim (--omp-barrier-elim)..." >&2
    BARRIER_ELIM_FLAG="--omp-barrier-elim"
    compile_opt "$src" "$d" "${name}_ab_elim" on >&2 \
        || { echo "  ERROR elim compile" >&2; BARRIER_ELIM_FLAG=""; emit_na "$name"; return; }
    BARRIER_ELIM_FLAG=""

    local out
    out=$(run_benchmark_ab "$d/${name}_ab_base" "$d/${name}_ab_elim" \
                           "$THREADS" "$d/ab_base.log" "$d/ab_elim.log") \
        || { emit_na "$name"; return; }
    local C_BASE SD_BASE C_ELIM SD_ELIM
    read -r C_BASE SD_BASE C_ELIM SD_ELIM <<< "$out"

    local DELTA DELTA_SD
    read -r DELTA DELTA_SD < <(awk -v b="$C_BASE" -v e="$C_ELIM" \
        -v sb="$SD_BASE" -v se="$SD_ELIM" 'BEGIN {
            if (b + 0 == 0) { print "NA NA"; exit }
            printf "%.4f %.4f", 100 * (b - e) / b, 100 * sqrt(sb*sb + se*se) / b
        }')

    {
        echo ""
        printf "  %-24s %18s\n" "base (${THREADS}T) cycles" "$C_BASE"
        printf "  %-24s %18s\n" "elim (${THREADS}T) cycles" "$C_ELIM"
        printf "  %-24s %17s%%\n" "risparmio" "${DELTA} ± ${DELTA_SD}"
        echo ""
    } >&2

    echo "${name};${C_BASE};${SD_BASE};${C_ELIM};${SD_ELIM};${DELTA};${DELTA_SD}" >> "$CSV"
    prune_kernel "$d"
}

run_kernel_pulp_ab() {
    local src="$1" name="$2"
    local d="$OUTDIR/$name/performance"
    mkdir -p "$d"

    echo -e "${BOLD}── $name${RESET}" >&2

    # pulp.sh splices $BARRIER_ELIM_FLAG into the mlir-opt-omp command, exactly
    # as native.sh does, so the flag is set around each build and cleared after.
    local base elim
    echo "  [1/2] base (senza la pass)..." >&2
    : > "$d/ab_base.log"
    BARRIER_ELIM_FLAG=""
    if ! base="$(pulp_cell "$src" opt_par "$d" "$d/ab_base.log")"; then
        echo "  ERROR: base failed" >&2; emit_na "$name"; return
    fi
    echo "  [2/2] elim (--omp-barrier-elim)..." >&2
    : > "$d/ab_elim.log"
    BARRIER_ELIM_FLAG="--omp-barrier-elim"
    if ! elim="$(pulp_cell "$src" opt_par "$d" "$d/ab_elim.log")"; then
        echo "  ERROR: elim failed" >&2; BARRIER_ELIM_FLAG=""; emit_na "$name"; return
    fi
    BARRIER_ELIM_FLAG=""

    local C_BASE="${base%%;*}" C_ELIM="${elim%%;*}"
    if [ "$C_BASE" = "NA" ] || [ "$C_ELIM" = "NA" ]; then
        echo -e "    ${YELLOW}WARNING: no 'Cycles =' line — see $d/ab_*.log${RESET}" >&2
        emit_na "$name"; return
    fi

    # Positive = the pass saved time. No uncertainty to propagate here.
    local DELTA
    DELTA=$(awk -v b="$C_BASE" -v e="$C_ELIM" 'BEGIN {
        if (b + 0 == 0) { print "NA"; exit }
        printf "%.4f", 100 * (b - e) / b
    }')

    {
        echo ""
        printf "  %-24s %18s\n" "base cycles" "$C_BASE"
        printf "  %-24s %18s\n" "elim cycles" "$C_ELIM"
        printf "  %-24s %17s%%\n" "risparmio" "$DELTA"
        echo ""
    } >&2

    echo "${name};${C_BASE};0;${C_ELIM};0;${DELTA};0" >> "$CSV"
    prune_kernel "$d"
}

prune_kernel() {   # $1 = the kernel's output directory
    [ -n "${KEEP:-}" ] && return 0
    rm -rf "$1"
}

run_kernel() {
    local kernel="$1" src
    if ! src="$(resolve_src "$kernel")"; then
        echo "  SKIP (not found): $kernel" >&2
        return
    fi
    local name; name="$(basename "${src%-omp.c}")-omp"

    if [ "$TARGET" = "pulp" ]; then
        if [ "$BARRIER_ELIM" = "both" ]; then
            run_kernel_pulp_ab "$src" "$name"
        else
            run_kernel_pulp "$src" "$name"
        fi
        return
    fi
    if [ "$BARRIER_ELIM" = "both" ]; then
        run_kernel_ab "$src" "$name"
        return
    fi

    local d="$OUTDIR/$name/performance"
    mkdir -p "$d"

    echo -e "${BOLD}── $name${RESET}" >&2

    # --- compile the four cells --------------------------------------------
    echo "  [1/4] ref_seq (native, no omp)..." >&2
    compile_ref "$src" "$d" "${name}_ref_seq" off >&2 \
        || { echo "  ERROR ref_seq compile" >&2; emit_na "$name"; return; }
    echo "  [2/4] ref_par (native, omp)..." >&2
    compile_ref "$src" "$d" "${name}_ref_par" on  >&2 \
        || { echo "  ERROR ref_par compile" >&2; emit_na "$name"; return; }
    echo "  [3/4] opt_seq (our tool, no omp)..." >&2
    compile_opt "$src" "$d" "${name}_opt_seq" off >&2 \
        || { echo "  ERROR opt_seq compile" >&2; emit_na "$name"; return; }
    echo "  [4/4] opt_par (our tool, omp)..." >&2
    compile_opt "$src" "$d" "${name}_opt_par" on  >&2 \
        || { echo "  ERROR opt_par compile" >&2; emit_na "$name"; return; }

    # --- time them ---------------------------------------------------------
    local C_REF_SEQ C_REF_PAR C_OPT_SEQ C_OPT_PAR
    C_REF_SEQ=$(run_benchmark "$d/${name}_ref_seq" 1         ref_seq "$d/ref_seq.log") || { emit_na "$name"; return; }
    C_REF_PAR=$(run_benchmark "$d/${name}_ref_par" "$THREADS" ref_par "$d/ref_par.log") || { emit_na "$name"; return; }
    C_OPT_SEQ=$(run_benchmark "$d/${name}_opt_seq" 1         opt_seq "$d/opt_seq.log") || { emit_na "$name"; return; }
    C_OPT_PAR=$(run_benchmark "$d/${name}_opt_par" "$THREADS" opt_par "$d/opt_par.log") || { emit_na "$name"; return; }

    # --- derived metrics ---------------------------------------------------
    local SP_NATIVE SP_OPT OPT_VS_NAT_PAR OPT_VS_NAT_SEQ
    SP_NATIVE=$(ratio "$C_REF_SEQ" "$C_REF_PAR")
    SP_OPT=$(ratio    "$C_OPT_SEQ" "$C_OPT_PAR")
    OPT_VS_NAT_PAR=$(ratio "$C_REF_PAR" "$C_OPT_PAR")
    OPT_VS_NAT_SEQ=$(ratio "$C_REF_SEQ" "$C_OPT_SEQ")

    # --- per-kernel console summary ----------------------------------------
    {
        echo ""
        printf "  %-22s %18s %18s\n" "" "native (ref)" "our tool (opt)"
        printf "  %-22s %18s %18s\n" "seq (1T) cycles"  "$C_REF_SEQ" "$C_OPT_SEQ"
        printf "  %-22s %18s %18s\n" "par (${THREADS}T) cycles" "$C_REF_PAR" "$C_OPT_PAR"
        printf "  %-22s %18s %18s\n" "self speedup seq→par" "${SP_NATIVE}x" "${SP_OPT}x"
        printf "  %-22s %18s\n" "opt vs native (par)" "${OPT_VS_NAT_PAR}x"
        printf "  %-22s %18s\n" "opt vs native (seq)" "${OPT_VS_NAT_SEQ}x"
        echo ""
    } >&2

    echo "${name};${C_REF_SEQ};${C_REF_PAR};${C_OPT_SEQ};${C_OPT_PAR};${SP_NATIVE};${SP_OPT};${OPT_VS_NAT_PAR};${OPT_VS_NAT_SEQ}" >> "$CSV"
    prune_kernel "$d"
}

emit_na() {
    local row="$1;NA;NA;NA;NA;NA;NA;NA;NA"
    if [ "$BARRIER_ELIM" = "both" ]; then
        row="$1;NA;NA;NA;NA;NA;NA"
    elif [ "$TARGET" = "pulp" ]; then
        row="$row;NA;NA;NA;NA"
    fi
    echo "$row" >> "$CSV"
}

plot_suffix() {
    local sel
    if [ -n "${KERNELS:-}" ]; then
        local k parts=()
        for k in $KERNELS; do
            k="${k##*/}"
            parts+=("${k%.c}")
        done
        sel=$(IFS=_; printf '%s' "${parts[*]}")
    else
        sel="suite"
    fi
    local ds="${DATASET%_DATASET}"
    printf '%s_%s' "$sel" "${ds,,}"
}

render_plot() {   # $1 = metric (speedup | size), default speedup
    local metric="${1:-speedup}"
    local script="$SCRIPT_DIR/lib/plot_speedup.py"
    local tag=""; [ "$metric" = "size" ] && tag="_size"
    local png="$OUTDIR/results_performance_$(plot_suffix)$BARRIER_ELIM_FILE_TAG$tag.png"
    local py; py="$(plot_python)" || return
    echo -e "${CYAN}[plot] rendering $metric chart...${RESET}" >&2
    if "$py" "$script" "$CSV" "$png" --runtime "$RUNTIME" --metric "$metric"; then
        echo "  Plot  — $png"
    else
        echo -e "${YELLOW}[plot] plot_speedup.py failed — see output above${RESET}" >&2
    fi
}

# The A/B chart: what the pass saved on each kernel, with the error bar that
# says whether the sign can be trusted. Same best-effort contract as above.
render_ab_plot() {
    local script="$SCRIPT_DIR/lib/plot_delta.py"
    local png="$OUTDIR/results_performance_$(plot_suffix)$BARRIER_ELIM_FILE_TAG.png"
    local py; py="$(plot_python)" || return
    echo -e "${CYAN}[plot] rendering A/B chart...${RESET}" >&2
    if "$py" "$script" "$CSV" "$png" --runtime "$RUNTIME" --threads "$THREADS"; then
        echo "  Plot  — $png"
    else
        echo -e "${YELLOW}[plot] plot_delta.py failed — see output above${RESET}" >&2
    fi
}

# --- Main ------------------------------------------------------------------
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_performance$BARRIER_ELIM_FILE_TAG.csv"
CSV_HEADER="kernel;ref_seq_cyc;ref_par_cyc;opt_seq_cyc;opt_par_cyc;speedup_native;speedup_opt;opt_vs_native_par;opt_vs_native_seq"
if [ "$BARRIER_ELIM" = "both" ]; then
    CSV_HEADER="kernel;base_par_cyc;base_sd;elim_par_cyc;elim_sd;delta_pct;delta_sd_pct"
elif [ "$TARGET" = "pulp" ]; then
    CSV_HEADER="$CSV_HEADER;size_ref_seq;size_ref_par;size_opt_seq;size_opt_par"
fi

echo "=== MLIR OpenMP PERFORMANCE COMPARISON ==="
if [ "$BARRIER_ELIM" = "both" ]; then
    echo "mode: A/B barrier elimination — our two builds against each other"
    echo "runtime: $RUNTIME    dataset: $DATASET    par threads: $THREADS"
    if [ "$TARGET" = "pulp" ]; then
        echo "timer: 'Cycles =' from the gvsoc log — 1 run per build, no alternation"
        echo "       (the simulator is deterministic: REPS would return the same number)"
    else
        echo "reps: $REPS (alternati base/elim)    timer: cycle-accurate TSC"
    fi
    echo "no native comparison: this run answers what the pass costs, nothing else"
    echo "polybench: $POLYBENCH"
    echo "rules: $RULES"
elif [ "$TARGET" = "pulp" ]; then
    echo "runtime: $RUNTIME (pulp/$PULP_PLATFORM)    dataset: $DATASET"
    echo "app dir: $PULP_APP_DIR"
    echo "ref: pulp-sdk gcc (make / OMP_NATIVE=1)    opt: CIR/MLIR kernel.o (OMP_OPT=1)"
    echo "timer: 'Cycles =' from the gvsoc log — 1 run/cell (simulator is deterministic)"
    echo "polybench: $POLYBENCH"
    echo "rules: $RULES"
else
    echo "runtime: $RUNTIME    dataset: $DATASET    par threads: $THREADS"
    echo "ref cc : $REF_CC    reps: $REPS    timer: cycle-accurate TSC"
    echo "polybench: $POLYBENCH"
    echo "rules: $RULES"
fi
echo ""

echo "$CSV_HEADER" > "$CSV"

select_kernels

if [ $# -ge 1 ]; then
    KERNELS="$1"   # so plot_suffix names the kernel rather than the suite
    run_kernel "$1"
else
    for k in "${KERNEL_LIST[@]}"; do
        run_kernel "$k"
    done
fi

# Section 4.5 gives the saving, how many kernels improve, and the two ends of
# the per-kernel range. All four come out of the A/B CSV, so they can be read
# against the sentence that reports them rather than eyeballed.
compare_ab_to_claims() {
    local rows=0 bad=0 value tol verdict
    local best_k best worst_k worst
    read -r best_k best worst_k worst < <(awk -F';' '
        NR > 1 && $1 != "TOTAL" && $2 != "NA" {
            d = $6 + 0
            if (++n == 1 || d > hi) { hi = d; hk = $1 }
            if (n == 1 || d < lo) { lo = d; lk = $1 }
        }
        END { if (n) printf "%s %.4f %s %.4f", hk, hi, lk, lo
              else printf "- NA - NA" }' "$CSV")

    echo ""
    echo -e "${BOLD}  === AGAINST SECTION 4.5 (reference/claims.csv) ===${RESET}"
    printf '  %-26s %9s %9s   %s\n' quantity measured paper verdict

    check_one() {   # $1 metric  $2 subject  $3 label  $4 measured
        # cleared first: read leaves them untouched when claim_row finds no
        # row, and the previous quantity's numbers would be compared again.
        value=""; tol=""
        read -r value tol < <(claim_row "$1" "$2")
        [ -z "${value:-}" ] && return 0
        [ "$4" = "NA" ] && return 0
        rows=$((rows + 1))
        verdict="$(claim_verdict "$4" "$value" "${tol:-0}")"
        [ "$verdict" = "DIFFERS" ] && bad=$((bad + 1))
        printf '  %-26s %9s %9s   %s\n' "$3" "$4" "$value" "$verdict"
    }

    check_one barrier_saving_pct suite   "saving over the suite %" "$T_DELTA"
    check_one kernels_improving  suite   "kernels improving"       "$N_SIG"
    check_one barrier_delta_pct  best    "best kernel %"           "$best"
    check_one barrier_delta_pct  worst   "worst kernel %"          "$worst"
    unset -f check_one

    [ "$rows" -eq 0 ] && { echo "  no rows for this run in $CLAIMS"; return 0; }
    echo ""
    echo "  Best here is $best_k, worst is $worst_k. The paper names trisolv at"
    echo "  the top of the range; a different kernel there is not a regression,"
    echo "  but the sentence in section 4.5 then names the wrong one."
    if [ "$bad" -ne 0 ]; then
        echo "  $bad of $rows differ — the gvsoc numbers are exact, so a gap is"
        echo "  a real change in what we emit or a stale sentence, not noise."
    fi
}

if [ "$BARRIER_ELIM" = "both" ]; then
    read -r T_BASE T_ELIM T_DELTA N_OK N_SIG < <(awk -F';' -v det="$([ "$TARGET" = "pulp" ] && echo 1 || echo 0)" '
        NR == 1 || $2 == "NA" { next }
        {
            b += $2; e += $4; n++
            if (det) { if ($6 + 0 > 0) sig++ }
            else if ($6 + 0 > 2 * ($7 + 0)) sig++
        }
        END {
            if (n == 0) { print "NA NA NA 0 0"; exit }
            printf "%.0f %.0f %.4f %d %d", b, e, 100 * (b - e) / b, n, sig
        }' "$CSV")

    echo "TOTAL;${T_BASE};;${T_ELIM};;${T_DELTA};" >> "$CSV"

    echo ""
    echo -e "${BOLD}=== SUMMARY A/B (${DATASET}, ${THREADS}T, runtime=${RUNTIME}) ===${RESET}"
    printf "${BOLD}  %-20s %16s %16s %12s${RESET}\n" "kernel" "base" "elim" "risparmio"
    if [ "$TARGET" = "pulp" ]; then
        awk -F';' 'NR > 1 && $1 != "TOTAL" && $2 != "NA" {
            printf "  %-20s %16.0f %16.0f %10s%%\n", $1, $2, $4, $6
        }' "$CSV"
    else
        awk -F';' 'NR > 1 && $1 != "TOTAL" && $2 != "NA" {
            printf "  %-20s %16.0f %16.0f %10s%% ± %s\n", $1, $2, $4, $6, $7
        }' "$CSV"
    fi
    echo "  --------------------------------------------------------------------"
    printf "${BOLD}  %-20s %16s %16s %10s%%${RESET}\n" "TOTALE" "$T_BASE" "$T_ELIM" "$T_DELTA"
    echo ""
    if [ "$TARGET" = "pulp" ]; then
        echo "  $N_SIG kernel su $N_OK migliorano."
        echo "  Nessuna barra d'errore: il simulatore è deterministico, quindi ogni"
        echo "  segno è esatto — ma è un'unica misura, non una media."
    else
        echo "  $N_SIG kernel su $N_OK con un risparmio maggiore del doppio del proprio errore."
        echo "  Il resto è sotto la sensibilità della misura, non necessariamente zero."
    fi

    # --- against the paper ---------------------------------------------------
    # Section 4.5 states this run's four aggregates. They are on GAP8 and over
    # the whole suite, so anything else is not comparable with them.
    if [ "$TARGET" = "pulp" ] && [ $# -eq 0 ] \
       && [ "${#KERNEL_LIST[@]}" -eq "${#ALL_KERNELS[@]}" ] \
       && [ -f "$CLAIMS" ]; then
        compare_ab_to_claims
    fi

    echo ""
    echo "  Done — $CSV"
    if is_true "$PLOT"; then render_ab_plot; fi
    exit 0
fi

read -r GM_NATIVE GM_OPT GM_VS_PAR GM_VS_SEQ < <(awk -F';' '
    NR == 1 { next }                         # header
    $6 == "NA" { next }                      # skipped/failed kernel
    { ln++; gn += log($6); go += log($7); gp += log($8); gs += log($9) }
    END {
        if (ln == 0) { print "NA NA NA NA"; exit }
        printf "%.4f %.4f %.4f %.4f",
               exp(gn/ln), exp(go/ln), exp(gp/ln), exp(gs/ln)
    }' "$CSV")

echo "GEOMEAN;;;;;${GM_NATIVE};${GM_OPT};${GM_VS_PAR};${GM_VS_SEQ}" >> "$CSV"

# --- Console table ---------------------------------------------------------
echo ""
if [ "$TARGET" = "pulp" ]; then
    SUMMARY_LABEL="${DATASET}, ${PULP_PLATFORM}, runtime=${RUNTIME}"
else
    SUMMARY_LABEL="${DATASET}, ${THREADS}T, runtime=${RUNTIME}"
fi
echo -e "${BOLD}=== SUMMARY (${SUMMARY_LABEL}) ===${RESET}"
printf "${BOLD}  %-20s %10s %10s %12s %12s${RESET}\n" \
    "kernel" "nat s→p" "opt s→p" "opt/nat par" "opt/nat seq"
awk -F';' 'NR > 1 && $1 != "GEOMEAN" {
    printf "  %-20s %9sx %9sx %11sx %11sx\n", $1, $6, $7, $8, $9
}' "$CSV"
echo "  --------------------------------------------------------------------"
printf "${BOLD}  %-20s %9sx %9sx %11sx %11sx${RESET}\n" \
    "GEOMEAN" "$GM_NATIVE" "$GM_OPT" "$GM_VS_PAR" "$GM_VS_SEQ"
echo ""
echo "  Legend (geomean = geometric mean across kernels):"
echo "    nat s→p      native  seq/par  — native compiler's own parallel scaling"
echo "    opt s→p      opt     seq/par  — this tool's own parallel scaling"
echo "    opt/nat par  ref_par/opt_par  — parallel runtime, native vs this tool"
echo "    opt/nat seq  ref_seq/opt_seq  — sequential runtime, native vs this tool"
echo "    A ratio > 1.0 means lower runtime (faster); < 1.0 means higher (slower)."
echo ""
echo "  Done — $CSV"

# --- Optional plots ---------------------------------------------------------
if is_true "$PLOT"; then
    render_plot
    if [ "$TARGET" = "pulp" ]; then render_plot size; fi
fi

if is_true "${COMPARE:-true}" && [ "$BARRIER_ELIM" != "both" ]; then
    compare_py="${PLOT_PYTHON:-}"
    [ -n "$compare_py" ] || compare_py="$(command -v python3 || command -v python)" || compare_py=""
    if [ -n "$compare_py" ]; then
        # The configuration goes with the CSV: the comparison only runs
        # when this run was made the way the figure was, and it is the one
        # that knows which configuration that is.
        "$compare_py" "$SCRIPT_DIR/lib/compare_to_reference.py" \
            "$CSV" --runtime "$RUNTIME" \
            --dataset "$DATASET" --threads "$THREADS" \
            --barrier-elim "$BARRIER_ELIM" || true
    fi
fi
exit 0
