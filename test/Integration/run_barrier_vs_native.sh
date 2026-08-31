#!/bin/bash
# run_barrier_vs_native.sh — counts team-barrier call sites: ours with and
# without --omp-barrier-elim, against clang's and gcc's on the same kernels.
#
# The three LLVM columns are iomp and are counted in LLVM IR after -O3, so both
# sides are measured at the same stage. gcc gets its own column counted at -O0,
# because from -O1 it spreads the same barriers over more call sites.
#
# Usage:
#   ./run_barrier_vs_native.sh                       # kernels in the suite
#   ./run_barrier_vs_native.sh path/to/kernel-omp.c
#   VERBOSE=1 ./run_barrier_vs_native.sh             # let the tools report why
#
# RUNTIME=iomp only. Leaves results_barrier_vs_native.csv.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Nothing is executed and nothing is built for the target.
SKIP_PULP_SDK=1

if [ -n "${RUNTIME:-}" ] && [ "$RUNTIME" != "iomp" ]; then
    echo "ERROR: RUNTIME=$RUNTIME — this comparison only runs on iomp, see the header." >&2
    exit 2
fi
RUNTIME=iomp
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

# No dumps and no timer: nothing here is run, and both sides must see the same
# defines or they would not be compiling the same program.
POLYBENCH_CFLAGS="$POLYBENCH_ROOT_CFLAGS"

OUTDIR="${OUTDIR:-$PWD/results}/$RUNTIME"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_barrier_vs_native.csv"
ARTDIR="$OUTDIR/barrier_vs_native"
rm -rf "$ARTDIR"
mkdir -p "$ARTDIR"

ERRSINK="/dev/null"
[ -n "${VERBOSE:-}" ] && ERRSINK="/dev/stderr"

build_ours() {
    local src="$1" out="$2" flag="$3"
    local name; name="$(basename "${src%.c}")"

    "$CLANG" -S $CLANG_STRICT_FP $WARN_SUPPRESS \
        -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC" -I"$(dirname "$src")" -I"$INC_OMP" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" -o "$TMPDIR/$name.cir" 2>"$ERRSINK" || return 1
    "$CIR_OPT" "$TMPDIR/$name.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$TMPDIR/$name-s1.mlir" 2>"$ERRSINK" || return 1
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$TMPDIR/$name-s1.mlir"
    "$MLIR_OPT_OMP" --allow-unregistered-dialect \
        --omp-lower-dsl="$RULES" --omp-lower-runtime="$RUNTIME" \
        ${flag:+$flag} \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        "$TMPDIR/$name-s1.mlir" > "$TMPDIR/$name-s2.mlir" 2>"$ERRSINK" || return 1
    "$MLIR_OPT" "$TMPDIR/$name-s2.mlir" \
        --canonicalize --cse --sccp --symbol-dce \
        --loop-invariant-code-motion --canonicalize --cse \
        --convert-arith-to-llvm --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$TMPDIR/$name-s3.mlir" 2>"$ERRSINK" || return 1
    "$MLIR_TRANSLATE" "$TMPDIR/$name-s3.mlir" --mlir-to-llvmir \
        > "$TMPDIR/$name.ll" 2>"$ERRSINK" || return 1
    "$OPT" -S -O3 "$TMPDIR/$name.ll" > "$out" 2>"$ERRSINK" || return 1
}

# Post-O3 the call prints as `tail call void @__kmpc_barrier(...)`, so match the
# callee and count occurrences rather than lines.
count_barriers() {   # $1 = .ll file
    grep -o 'call void @__kmpc_barrier' "$1" | wc -l
}

build_gcc() {   # $1 = source, $2 = output .s
    "$GCC" -fopenmp -O0 -S $GCC_STRICT_FP \
        -I"$INC" -I"$(dirname "$1")" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$1" -o "$2" 2>"$ERRSINK"
}

count_gcc_barriers() {   # $1 = .s file
    grep -cE '\b(call|jmp)\b.*GOMP_barrier' "$1" || true
}

# How the kernel spells its parallel loop. clang elides the trailing barrier
# only for the combined directive, so this column is what explains the delta.
pragma_form() {   # $1 = source
    if grep -qE '#pragma omp parallel[[:space:]]+for' "$1"; then
        echo "combined"
    else
        echo "split"
    fi
}

if [ $# -gt 0 ]; then
    KERNEL_LIST=("$@")
else
    select_kernels
fi

echo "=== TEAM BARRIERS: OURS vs CLANG (both after -O3) ==="
echo "runtime:   $RUNTIME (fixed)    dataset: $DATASET"
echo "polybench: $POLYBENCH"
echo "kernels:   ${#KERNEL_LIST[@]}"
echo "files:     $ARTDIR"
echo

echo "kernel,pragma_form,clang,gcc_o0,ours_baseline,ours_elim,saved_vs_clang" > "$CSV"
printf '  %-22s %-9s %6s %6s %8s %8s %8s\n' kernel form clang gcc base elim 'vs clang'

# Intermediates only (.cir and the MLIR stages); the counted files go straight
# to $ARTDIR and stay there.
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

FAILED=0
T_CLANG=0; T_GCC=0; T_BASE=0; T_ELIM=0

for k in "${KERNEL_LIST[@]}"; do
    src="$(resolve_src "$k")" || { echo "  SKIP (not found): $k"; continue; }
    name="$(basename "$src" .c)"
    form="$(pragma_form "$src")"
    kdir="$ARTDIR/$name"
    mkdir -p "$kdir"

    "$CLANG" -fopenmp -O3 -S -emit-llvm $CLANG_STRICT_FP $WARN_SUPPRESS \
        -I"$INC" -I"$(dirname "$src")" -I"$INC_OMP" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" -o "$kdir/clang-O3.ll" 2>"$ERRSINK" \
        || { echo "  ERROR (clang): $name"; echo "$name,$form,,,,," >> "$CSV"; FAILED=1; continue; }

    build_ours "$src" "$kdir/ours-baseline-O3.ll" "" \
        || { echo "  ERROR (ours, baseline): $name"; echo "$name,$form,,,,," >> "$CSV"; FAILED=1; continue; }
    build_ours "$src" "$kdir/ours-elim-O3.ll" "--omp-barrier-elim" \
        || { echo "  ERROR (ours, barrier-elim): $name"; echo "$name,$form,,,,," >> "$CSV"; FAILED=1; continue; }

    g=""
    if build_gcc "$src" "$kdir/gcc-O0.s"; then
        g="$(count_gcc_barriers "$kdir/gcc-O0.s")"
        T_GCC=$((T_GCC + g))
    else
        rm -f "$kdir/gcc-O0.s"
    fi

    c="$(count_barriers "$kdir/clang-O3.ll")"
    b="$(count_barriers "$kdir/ours-baseline-O3.ll")"
    e="$(count_barriers "$kdir/ours-elim-O3.ll")"
    saved=$((c - e))

    T_CLANG=$((T_CLANG + c)); T_BASE=$((T_BASE + b)); T_ELIM=$((T_ELIM + e))
    printf '  %-22s %-9s %6s %6s %8s %8s %8s\n' \
        "$name" "$form" "$c" "${g:--}" "$b" "$e" "$saved"
    echo "$name,$form,$c,$g,$b,$e,$saved" >> "$CSV"
done

echo
echo "  clang: $T_CLANG    ours without the pass: $T_BASE    ours with it: $T_ELIM"
if [ "$T_CLANG" -gt 0 ]; then
    awk -v c="$T_CLANG" -v e="$T_ELIM" \
        'BEGIN { printf "  %d fewer than clang (%.1f%%)\n", c - e, 100 * (c - e) / c }'
fi
if [ "$T_GCC" -gt 0 ]; then
    echo "  gcc (before -O3): $T_GCC    ours with the pass: $T_ELIM"
    echo "    -O0 is the stage that measures the elision: gcc decides it in the"
    echo "    front-end, so -O0 already shows it, while from -O1 jump threading"
    echo "    splits the path where a thread's chunk comes out empty and copies"
    echo "    the barrier sequence into it (28 -> 43 over the full suite; -O3"
    echo "    -fno-thread-jumps puts every kernel back on its -O0 number). Say"
    echo "    which stage when quoting the column."
fi

echo
echo "  Counted in, one directory per kernel under"
echo "    $ARTDIR/<kernel>/"
echo "      clang-O3.ll  ours-baseline-O3.ll  ours-elim-O3.ll  gcc-O0.s"
echo "  Recount any row:"
echo "    grep -o 'call void @__kmpc_barrier' <file>.ll | wc -l"
echo "    grep -cE '\b(call|jmp)\b.*GOMP_barrier' gcc-O0.s"
echo "  Done — $CSV"

[ "$FAILED" -ne 0 ] && { echo "  some kernels failed (VERBOSE=1 for the tool output)"; exit 1; }
exit 0
