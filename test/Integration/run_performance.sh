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
# Timing uses PolyBench's cycle-accurate TSC timer. Each cell is run 5 times;
# the min and max are dropped and the mean of the 3 middle runs is reported,
# together with its standard deviation. A cell whose relative std-dev exceeds
# VARIANCE_ACCEPTED% is flagged (noisy machine / background load).
#
# All shared setup (config, tools, kernel lists, the compile pipeline) lives in
# common.sh — same config.env as run_correctness.sh.
#
# Usage:
#   ./run_performance.sh                              # all kernels, defaults
#   RUNTIME=libgomp ./run_performance.sh              # pick the runtime
#   DATASET=LARGE_DATASET THREADS=16 ./run_performance.sh
#   ./run_performance.sh path/to/kernel-omp.c         # a single kernel
#   SUITE=full POLYBENCH=/path/to/checkout ./run_performance.sh
#
# CSV output (under $OUTDIR, default ./results):
#   results_performance.csv      one row per kernel + a GEOMEAN summary row
#   <kernel>-omp/performance/...  the four binaries and their raw timing logs
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Perf wants a real workload by default; MINI is dominated by thread-spawn cost.
# DATASET / config.env still override this (see common.sh).
DATASET_DEFAULT="LARGE_DATASET"
# shellcheck source=common.sh
. "$SCRIPT_DIR/common.sh"

# --- Performance-specific config -------------------------------------------
THREADS="${THREADS:-16}"             # thread count for the parallel cells
REPS="${REPS:-5}"                    # timed runs per cell (>=3; min+max dropped)
VARIANCE_ACCEPTED="${VARIANCE_ACCEPTED:-5}"   # rel std-dev warn threshold (%)
OUTDIR="${OUTDIR:-$PWD/results}"

# Performance times the kernel with the cycle-accurate TSC timer. This is
# mutually exclusive with -DPOLYBENCH_DUMP_ARRAYS, hence its own CFLAGS.
POLYBENCH_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_CYCLE_ACCURATE_TIMER $POLYBENCH_ROOT_CFLAGS"

# --- Stats helpers ---------------------------------------------------------
# run_benchmark <binary> <nthreads> <label> <logfile>
# Runs the binary REPS times, drops min+max, and computes the mean / std-dev /
# min over the middle runs. Echoes "mean" on stdout; also sets BENCH_STDDEV,
# BENCH_MIN, BENCH_RELSD (relative std-dev %). Returns non-zero on failure.
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

    # Keep the middle (REPS-2) values after sorting numerically.
    local keep=$((REPS - 2))
    [ "$keep" -lt 1 ] && keep=$REPS   # REPS<3: keep them all
    local stats
    stats=$(sort -g "$logfile" | head -n $((REPS == keep ? REPS : REPS - 1)) \
            | tail -n "$keep" | awk '
        /^[0-9.eE+-]+$/ { v[++n] = $1; s += $1; if (n == 1 || $1 < mn) mn = $1 }
        END {
            if (n < 1) { print "ERR"; exit }
            m = s / n
            for (i = 1; i <= n; i++) { d = v[i] - m; ss += d * d }
            sd = sqrt(ss / n)
            rsd = (m != 0) ? sd / m * 100 : 0
            printf "%.6f %.6f %.6f %.4f", m, sd, mn, rsd
        }')

    if [ "$stats" = "ERR" ] || [ -z "$stats" ]; then
        echo -e "    ${RED}no numeric output — is -DPOLYBENCH_TIME set?${RESET}" >&2
        return 1
    fi

    local mean
    read -r mean BENCH_STDDEV BENCH_MIN BENCH_RELSD <<< "$stats"

    local warn
    warn=$(awk -v r="$BENCH_RELSD" -v t="$VARIANCE_ACCEPTED" \
        'BEGIN { print (r < t) ? "ok" : "warn" }')
    if [ "$warn" = "warn" ]; then
        echo -e "    ${YELLOW}[noisy]${RESET} ${label}: rel std-dev ${BENCH_RELSD}% > ${VARIANCE_ACCEPTED}%" >&2
    fi
    echo -e "    ${BOLD}${label}: ${mean} cyc${RESET} (sd ${BENCH_STDDEV}, min ${BENCH_MIN})" >&2

    printf "%s" "$mean"
}

# ratio <a> <b> -> a/b with 4 decimals, or NA.
ratio() {
    awk -v a="$1" -v b="$2" \
        'BEGIN { if (b+0 == 0) print "NA"; else printf "%.4f", a / b }'
}

# --- Per-kernel driver -----------------------------------------------------
run_kernel() {
    local kernel="$1" src
    if ! src="$(resolve_src "$kernel")"; then
        echo "  SKIP (not found): $kernel" >&2
        return
    fi
    local name; name="$(basename "${src%-omp.c}")-omp"
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
    echo "$1;NA;NA;NA;NA;NA;NA;NA;NA" >> "$CSV"
}

# --- Main ------------------------------------------------------------------
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_performance.csv"
CSV_HEADER="kernel;ref_seq_cyc;ref_par_cyc;opt_seq_cyc;opt_par_cyc;speedup_native;speedup_opt;opt_vs_native_par;opt_vs_native_seq"

echo "=== MLIR OpenMP PERFORMANCE COMPARISON ==="
echo "runtime: $RUNTIME    dataset: $DATASET    par threads: $THREADS    suite: $SUITE"
echo "ref cc : $REF_CC    reps: $REPS    timer: cycle-accurate TSC"
echo "polybench: $POLYBENCH"
echo "rules: $RULES"
echo ""

echo "$CSV_HEADER" > "$CSV"

select_kernels

if [ $# -ge 1 ]; then
    run_kernel "$1"
else
    for k in "${KERNEL_LIST[@]}"; do
        run_kernel "$k"
    done
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
echo -e "${BOLD}=== SUMMARY (${DATASET}, ${THREADS}T, runtime=${RUNTIME}) ===${RESET}"
printf "${BOLD}  %-20s %10s %10s %12s %12s${RESET}\n" \
    "kernel" "nat s→p" "opt s→p" "opt/nat par" "opt/nat seq"
awk -F';' 'NR > 1 && $1 != "GEOMEAN" {
    printf "  %-20s %9sx %9sx %11sx %11sx\n", $1, $6, $7, $8, $9
}' "$CSV"
echo "  --------------------------------------------------------------------"
printf "${BOLD}  %-20s %9sx %9sx %11sx %11sx${RESET}\n" \
    "GEOMEAN" "$GM_NATIVE" "$GM_OPT" "$GM_VS_PAR" "$GM_VS_SEQ"
echo ""
echo "  opt/nat par > 1  ⇒  our parallel code beats the native compiler."
echo "Done — $CSV"
