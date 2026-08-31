#!/bin/bash
# =============================================================================
# run_performance.sh — MLIR OpenMP performance comparison (our tool vs native)
#
# For each PolyBench kernel it builds the 2x2 matrix and times each cell:
#
#                 sequential (1 thread)      parallel ($THREADS threads)
#   native (ref)  ref_seq  (clang/gcc -O3)   ref_par  (clang/gcc -fopenmp)
#   our tool(opt) opt_seq  (CIR/MLIR, no omp) opt_par  (CIR/MLIR -fopenmp)
#
# From those four numbers it reports:
#   speedup_native      = ref_seq / ref_par   native self seq->par speedup
#   speedup_opt         = opt_seq / opt_par    our    self seq->par speedup
#   opt_vs_native_par   = ref_par / opt_par    headline: our parallel vs the
#                                              native parallel compiler
#                                              (>1 => our tool is faster)
#   opt_vs_native_seq   = ref_seq / opt_seq    same, sequential
#
# Timing uses PolyBench's cycle-accurate TSC timer. Each cell is run REPS times
# (10 by default); the min and max are dropped and the mean of the rest is
# reported, together with its standard deviation. A cell whose relative std-dev
# exceeds VARIANCE_ACCEPTED% is flagged (noisy machine / background load).
#
# RUNTIME=pmsis (PULP/gvsoc) builds the same 2x2 matrix but through the
# PolyBench-PULP harness Makefile (PULP_APP_DIR): ref = the PULP-SDK gcc
# (plain / OMP_NATIVE=1), opt = our riscv32 kernel.o (OMP_OPT=1). Cycles come
# from the 'Cycles = N' line in the gvsoc run log — one run per cell, the
# simulator is deterministic — and four binary-size columns are appended to
# the CSV. Only runs on machines with the GAP SDK + gvsoc installed.
#
# All shared setup (config, tools, kernel lists, the compile pipeline) lives in
# common.sh — same run.env as run_correctness.sh.
#
# Usage:
#   ./run_performance.sh                              # all kernels, defaults
#   RUNTIME=libgomp ./run_performance.sh              # pick the runtime
#   RUNTIME=pmsis ./run_performance.sh                # PULP/gvsoc (needs GAP SDK)
#   DATASET=LARGE_DATASET THREADS=16 ./run_performance.sh
#   ./run_performance.sh path/to/kernel-omp.c         # a single kernel
#   SUITE=full POLYBENCH=/path/to/checkout ./run_performance.sh
#   PLOT=true SUITE=full ./run_performance.sh          # also render the chart
#   BARRIER_ELIM=1 ./run_performance.sh                # with --omp-barrier-elim
#   BARRIER_ELIM=both ./run_performance.sh             # A/B: our two builds,
#                                                      # timed against each other
#
# Output lands under $OUTDIR/<runtime> (default ./results/<runtime>), one
# folder per runtime (iomp, libgomp, pmsis) so runs against different runtimes
# never overwrite each other. BARRIER_ELIM=1 adds a -barrier-elim suffix, so an
# optimised run keeps its own folder beside the baseline it is compared with:
#   results_performance.csv        one row per kernel + a GEOMEAN row
#   <kernel>-omp/performance/...   the four binaries and raw timing logs
#   results_performance_<sel>.png  speedup bar chart (only when PLOT=true;
#                                  needs python3 + matplotlib — see
#                                  plot_speedup.py). On RUNTIME=pmsis a second
#                                  chart, ..._size.png, plots the binary size
#                                  change from the size_* columns.
#                                  <sel> = the kernel
#                                  selection (the SUITE, full/bundled, or the
#                                  explicit kernel name(s)) + the dataset size,
#                                  e.g. _full_large or _gemm-omp_mini.
#
# The CSV and the chart carry a _barrier-elim suffix of their own under
# BARRIER_ELIM=1. The folder already says which run they came from, but these
# two are the files that leave it — into a paper, a slide, a mail — and there
# the folder is gone and the name is all that is left.
# =============================================================================

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
# Results are split per runtime and by BARRIER_ELIM — results/<runtime>/ and
# results/<runtime>-barrier-elim/ — so a run only ever replaces one of its own
# kind, and a baseline survives the optimised run it is compared against.
OUTDIR="${OUTDIR:-$PWD/results}/$RUNTIME$BARRIER_ELIM_DIR_TAG"
PLOT="${PLOT:-false}"                # true -> render a speedup bar chart at the end

# Performance times the kernel with the cycle-accurate TSC timer. This is
# mutually exclusive with -DPOLYBENCH_DUMP_ARRAYS, hence its own CFLAGS.
POLYBENCH_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_CYCLE_ACCURATE_TIMER $POLYBENCH_ROOT_CFLAGS"

# --- Stats helpers ---------------------------------------------------------
# run_benchmark <binary> <nthreads> <label> <logfile>
# Runs the binary REPS times, drops min+max, and computes the mean / std-dev /
# min over the middle runs. Echoes "mean" on stdout; also sets BENCH_STDDEV,
# BENCH_MIN, BENCH_RELSD (relative std-dev %). Returns non-zero on failure.

# bench_stats <logfile> -> "mean sd min relsd" over the middle REPS-2 values,
# nothing at all when the log holds no numbers. Split out of run_benchmark
# because the A/B timer below fills two logs and then wants the same maths on
# each of them.
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
# The A/B timer: one repetition of each binary in turn, rather than all of A
# and then all of B. Whatever the machine does during the run — a frequency
# step, another job arriving — lands on both sides within a second of each
# other instead of hours apart, which is the whole reason this mode exists.
# Echoes "meanA sdA meanB sdB" — everything the caller needs on stdout, since
# it runs inside a command substitution and anything it merely assigns would
# stay in the subshell.
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

# --- Per-kernel driver: pulp/gvsoc -----------------------------------------
# One gvsoc run per cell (build+run are fused in the harness 'make', and the
# simulator is deterministic, so REPS does not apply). Also records the size
# of the linked ELF for each cell.
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
}

# --- Per-kernel driver: BARRIER_ELIM=both ----------------------------------
# Our own two builds against each other, and nothing else: no native compiler,
# no sequential cells. Both would answer a different question, and the
# sequential build carries no barriers at all — it is compiled without -fopenmp
# — so it is identical in the two configurations by construction.
run_kernel_ab() {
    local src="$1" name="$2"
    local d="$OUTDIR/$name/performance"
    mkdir -p "$d"

    echo -e "${BOLD}── $name${RESET}" >&2

    # compile_opt splices $BARRIER_ELIM_FLAG into the mlir-opt-omp command, so
    # set it around each build. In this mode it is empty globally, and it is
    # put back empty afterwards: nothing else in the run should see it.
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

    # Positive = the pass saved time. The uncertainty carried alongside is what
    # says whether the sign means anything: propagating both std-devs, a delta
    # smaller than its own error bar is not a result.
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
}

# --- Per-kernel driver -----------------------------------------------------
run_kernel() {
    local kernel="$1" src
    if ! src="$(resolve_src "$kernel")"; then
        echo "  SKIP (not found): $kernel" >&2
        return
    fi
    local name; name="$(basename "${src%-omp.c}")-omp"

    if [ "$TARGET" = "pulp" ]; then
        run_kernel_pulp "$src" "$name"
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

# Suffix naming what this run covered, appended to the plot filename:
# the kernel selection — the SUITE name (full/bundled), or, when an explicit
# KERNELS list (or a single-kernel argument) was given, the kernel basenames
# (the part after the last '/', without the .c extension) joined by '_' —
# followed by the dataset size (LARGE_DATASET -> large).
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
        sel="$SUITE"
    fi
    local ds="${DATASET%_DATASET}"
    printf '%s_%s' "$sel" "${ds,,}"
}

# Render the speedup bar chart from $CSV via plot_speedup.py. Best-effort: a
# missing python/matplotlib is a warning, not a failure (the CSV is the result).
# plot_python (common.sh) picks the interpreter, the same way for every driver.
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

# gvsoc is deterministic and REPS does not apply there, so the drift this mode
# exists to cancel cannot happen: two ordinary runs already compare exactly.
if [ "$BARRIER_ELIM" = "both" ] && [ "$TARGET" = "pulp" ]; then
    echo "ERROR: BARRIER_ELIM=both is for the host runtimes." >&2
    echo "       On pmsis the simulator is deterministic — run it twice instead:" >&2
    echo "         RUNTIME=pmsis BARRIER_ELIM=0 ./run_performance.sh" >&2
    echo "         RUNTIME=pmsis BARRIER_ELIM=1 ./run_performance.sh" >&2
    exit 2
fi

echo "=== MLIR OpenMP PERFORMANCE COMPARISON ==="
if [ "$BARRIER_ELIM" = "both" ]; then
    echo "mode: A/B barrier elimination — our two builds against each other"
    echo "runtime: $RUNTIME    dataset: $DATASET    par threads: $THREADS    suite: $SUITE"
    echo "reps: $REPS (alternati base/elim)    timer: cycle-accurate TSC"
    echo "no native comparison: this run answers what the pass costs, nothing else"
    echo "polybench: $POLYBENCH"
    echo "rules: $RULES"
elif [ "$TARGET" = "pulp" ]; then
    echo "runtime: $RUNTIME (pulp/$PULP_PLATFORM)    dataset: $DATASET    suite: $SUITE"
    echo "app dir: $PULP_APP_DIR"
    echo "ref: pulp-sdk gcc (make / OMP_NATIVE=1)    opt: CIR/MLIR kernel.o (OMP_OPT=1)"
    echo "timer: 'Cycles =' from the gvsoc log — 1 run/cell (simulator is deterministic)"
    echo "polybench: $POLYBENCH"
    echo "rules: $RULES"
else
    echo "runtime: $RUNTIME    dataset: $DATASET    par threads: $THREADS    suite: $SUITE"
    echo "ref cc : $REF_CC    reps: $REPS    timer: cycle-accurate TSC"
    echo "polybench: $POLYBENCH"
    echo "rules: $RULES"
fi
echo ""

echo "$CSV_HEADER" > "$CSV"

select_kernels

if [ $# -ge 1 ]; then
    KERNELS="$1"   # so plot_suffix names the kernel, not the unused SUITE
    run_kernel "$1"
else
    for k in "${KERNEL_LIST[@]}"; do
        run_kernel "$k"
    done
fi

if [ "$BARRIER_ELIM" = "both" ]; then
    # --- A/B summary --------------------------------------------------------
    # Cycles summed rather than a geomean of ratios: the question here is how
    # much time the pass saves over the suite, and a ratio-geomean would give a
    # kernel that runs for a millisecond the same weight as one that runs for a
    # minute. The per-kernel percentages, each with its error bar, are in the
    # table above and in the CSV.
    read -r T_BASE T_ELIM T_DELTA N_OK N_SIG < <(awk -F';' '
        NR == 1 || $2 == "NA" { next }
        { b += $2; e += $4; n++; if ($6 + 0 > 2 * ($7 + 0)) sig++ }
        END {
            if (n == 0) { print "NA NA NA 0 0"; exit }
            printf "%.0f %.0f %.4f %d %d", b, e, 100 * (b - e) / b, n, sig
        }' "$CSV")

    echo "TOTAL;${T_BASE};;${T_ELIM};;${T_DELTA};" >> "$CSV"

    echo ""
    echo -e "${BOLD}=== SUMMARY A/B (${DATASET}, ${THREADS}T, runtime=${RUNTIME}) ===${RESET}"
    printf "${BOLD}  %-20s %16s %16s %12s${RESET}\n" "kernel" "base" "elim" "risparmio"
    awk -F';' 'NR > 1 && $1 != "TOTAL" && $2 != "NA" {
        printf "  %-20s %16.0f %16.0f %10s%% ± %s\n", $1, $2, $4, $6, $7
    }' "$CSV"
    echo "  --------------------------------------------------------------------"
    printf "${BOLD}  %-20s %16s %16s %10s%%${RESET}\n" "TOTALE" "$T_BASE" "$T_ELIM" "$T_DELTA"
    echo ""
    echo "  $N_SIG kernel su $N_OK con un risparmio maggiore del doppio del proprio errore."
    echo "  Il resto è sotto la sensibilità della misura, non necessariamente zero."
    echo ""
    echo "  Done — $CSV"
    if is_true "$PLOT"; then render_ab_plot; fi
    exit 0
fi

# --- Suite-level summary (geometric means over non-NA rows) ----------------
# Geomean is the standard way to summarise speedup ratios across a benchmark
# set (arithmetic mean over-weights the largest ratios).
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
# (an `if`, not `&&`: as the last command, a false PLOT must not turn into a
# non-zero exit code for the whole script)
if is_true "$PLOT"; then
    render_plot
    # Only the PULP path writes the size_* columns, and only there is the
    # footprint a result in its own right rather than a footnote — memory is
    # the binding constraint on the target, so it gets its own figure.
    if [ "$TARGET" = "pulp" ]; then render_plot size; fi
fi
