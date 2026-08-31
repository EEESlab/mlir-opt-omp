#!/bin/bash
# =============================================================================
# run_barrier_vs_native.sh — team barrier call sites, ours against the stock
# compiler, counted at the same stage
#
# This answers the question a reader asks: how many barriers are left compared
# with the compiler people actually use.
#
# The comparison is only worth anything if both sides are counted at the same
# point, so both are taken from LLVM IR *after* -O3:
#
#   clang     clang -fopenmp -O3 -S -emit-llvm
#   ours      the native.sh pipeline through step 6 (opt -S -O3)
#
# Same kernel, same -D$DATASET, same strict-FP flags on both. Counting our
# side in MLIR instead would compare pre-optimisation against post-, and any
# duplication -O3 performs would show up on one side only.
#
# Ours is built twice, with and without --omp-barrier-elim, so the table shows
# both what the pipeline emits on its own and what the pass takes off.
#
# The three LLVM columns are iomp: they speak one ABI (__kmpc_barrier), which
# is what makes a call count comparable between them.
#
# gcc is counted too, but in its own column and **before -O3**:
#
#   gcc       gcc -fopenmp -O0 -S, counting GOMP_barrier call sites
#
# because from -O1 on gcc spreads the same barriers over more call sites than
# the program needs. One pass accounts for all of it: jump threading splits the
# path where a thread's chunk comes out empty, and the split path carries its
# own copy of the barrier sequence. Over this suite the total goes 28 at -O0 to
# 43 at -O3, gemver alone 3 -> 7 — and `-O3 -fno-thread-jumps` puts every kernel
# back on its -O0 number exactly. That is what this stage choice rests on: -O0
# is not the convenient stage, it is the one that measures the elision instead
# of the CFG shape.
#
# (Not loop cloning: $GCC_STRICT_FP already turns the vectoriser off, so no
# kernel here gets a vector and a scalar copy.)
#
# The elision itself is decided in the front-end, so it is already applied at
# -O0 — and our own count does not move between stages (the same 59/25 in MLIR
# and after -O3), so the comparison holds even though the stages differ. State
# the stage when quoting it.
#
# Worth knowing what that column says: gcc performs this elision too, and
# reaches 27 where the pass reaches 25. Where clang misses it is the *split*
# spelling; where gcc misses it is a region holding a declaration, which puts
# the loop inside a block and out of reach of its check.
#
# Every file a number was counted in is kept, one directory per kernel, so the
# table can be checked instead of taken on trust:
#
#   results/iomp/barrier_vs_native/<kernel>/clang-O3.ll
#                                           ours-baseline-O3.ll
#                                           ours-elim-O3.ll
#                                           gcc-O0.s
#
# The counts are two greps over those files, printed at the end of the run.
#
# RUNTIME is pinned to iomp (see above) rather than read from run.env; the rest
# of the configuration — POLYBENCH, the tool paths, DATASET, KERNELS, OUTDIR —
# comes from there as usual.
#
# Usage:
#   ./run_barrier_vs_native.sh                       # kernels in the suite
#   ./run_barrier_vs_native.sh path/to/kernel-omp.c  # a single kernel
#   VERBOSE=1 ./run_barrier_vs_native.sh             # let the tools report why
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Nothing is executed and nothing is built for the target.
SKIP_PULP_SDK=1

# RUNTIME is not a knob here — the three LLVM columns compare only because they
# speak one ABI — so it is pinned before common.sh reads the config files. A
# run.env or RUN_ENV config sitting on another runtime is then ignored rather
# than turned into an error to work around: everything else those files carry
# (POLYBENCH, the tool paths, DATASET, KERNELS, OUTDIR) still applies, and that
# is what this script reads them for.
#
# An inline RUNTIME= is the one case worth refusing: that is someone asking for
# a comparison this script cannot make, so say so rather than quietly making a
# different one.
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
# The counted files are kept, not only the totals: a table of barrier counts is
# worth what recounting it is worth. Cleared first, so nothing a failed kernel
# left behind last time can be read as part of this run.
ARTDIR="$OUTDIR/barrier_vs_native"
rm -rf "$ARTDIR"
mkdir -p "$ARTDIR"

ERRSINK="/dev/null"
[ -n "${VERBOSE:-}" ] && ERRSINK="/dev/stderr"

# Our pipeline, native.sh steps 1-6: front-end through opt -O3. Stops before
# llc and the link, which the count does not need.
# $1 = source, $2 = output .ll, $3 = extra mlir-opt-omp flag or ""
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

# The gcc side, at -O0 and in assembly — see the header for why that stage.
# Anchoring on the call instruction keeps a .type or .globl line for the symbol
# out of the count.
build_gcc() {   # $1 = source, $2 = output .s
    "$GCC" -fopenmp -O0 -S $GCC_STRICT_FP \
        -I"$INC" -I"$(dirname "$1")" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$1" -o "$2" 2>"$ERRSINK"
}

# Tail calls count: from -O2 gcc emits the last barrier of a region as
# `jmp GOMP_barrier`, which an anchor on `call` alone drops. At -O0, the stage
# this column is taken at, there are none — but the regex has to be right for
# anyone who reruns it at another level.
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

    # gcc is the one column this comparison can do without: it answers a
    # different question at a different stage, so a machine with no usable gcc
    # leaves it blank rather than failing the run.
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
# On its own line, and never folded into the percentage above: gcc is the other
# question — not "do we beat the compiler people use" but "does a compiler that
# already does this reach the same place". Different stage, different claim.
if [ "$T_GCC" -gt 0 ]; then
    echo "  gcc (before -O3): $T_GCC    ours with the pass: $T_ELIM"
    echo "    -O0 is the stage that measures the elision: gcc decides it in the"
    echo "    front-end, so -O0 already shows it, while from -O1 jump threading"
    echo "    splits the path where a thread's chunk comes out empty and copies"
    echo "    the barrier sequence into it (28 -> 43 over the full suite; -O3"
    echo "    -fno-thread-jumps puts every kernel back on its -O0 number). Say"
    echo "    which stage when quoting the column."
fi

# Where the numbers came from. Printed on every run, not just on demand: a
# table of counts is only as good as the ability to recount it, and a reviewer
# should not have to read the script to find out how.
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
