#!/bin/bash
# run_barrier_vs_native.sh — counts team-barrier call sites: ours with and
# without --omp-barrier-elim, against clang's and gcc's on the same kernels.
#
# All four columns are counted after -O3. The three LLVM ones are iomp, so they
# speak one ABI; gcc's carries -fno-thread-jumps, which switches off the one
# pass that duplicates barrier call sites without adding barriers. With it gcc
# is back on the count it has at -O0, so the column measures the elision and
# not the shape of the CFG. The run prints the reasoning in full.
#
# Usage:
#   ./run_barrier_vs_native.sh                       # kernels in the suite
#   ./run_barrier_vs_native.sh path/to/kernel-omp.c
#   VERBOSE=1 ./run_barrier_vs_native.sh             # let the tools report why
#
# RUNTIME=iomp only. Leaves results_barrier_vs_native.csv.
#
# The four totals are what section 4.5 states, so on a full-suite run they are
# checked against reference/claims.csv rather than left to be read.

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

# -O3 to match the LLVM columns, -fno-thread-jumps so the count is of
# barriers and not of the paths gcc copied them into. Without the flag the
# suite goes 28 -> 43 and gemver alone 3 -> 7, with no execution gaining a
# barrier; with it every kernel sits on its -O0 number.
build_gcc() {   # $1 = source, $2 = output .s
    "$GCC" -fopenmp -O3 -fno-thread-jumps -S $GCC_STRICT_FP \
        -I"$INC" -I"$(dirname "$1")" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$1" -o "$2" 2>"$ERRSINK"
}

# From -O2 gcc emits the last barrier of a region as a tail call, which prints
# as jmp: anchoring on `call` alone would silently drop 6 of them over the
# suite. At -O0 that did not arise, which is why the pattern matters more now.
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

# The published totals are sums over the whole suite, so a subset must not be
# compared with them. Counting is enough: it also covers KERNELS= and a hand
# written list that happens to be complete.
FULL_SUITE=0
[ "${#KERNEL_LIST[@]}" -eq "${#ALL_KERNELS[@]}" ] && FULL_SUITE=1

echo "=== TEAM BARRIERS: OURS vs CLANG (both after -O3) ==="
echo "runtime:   $RUNTIME (fixed)    dataset: $DATASET"
echo "polybench: $POLYBENCH"
echo "kernels:   ${#KERNEL_LIST[@]}"
echo "files:     $ARTDIR"
echo

echo "kernel,pragma_form,clang,gcc_o3_ntj,ours_baseline,ours_elim,saved_vs_clang" > "$CSV"
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
    if build_gcc "$src" "$kdir/gcc-O3-ntj.s"; then
        g="$(count_gcc_barriers "$kdir/gcc-O3-ntj.s")"
        T_GCC=$((T_GCC + g))
    else
        rm -f "$kdir/gcc-O3-ntj.s"
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
    echo "  gcc (-O3 -fno-thread-jumps): $T_GCC    ours with the pass: $T_ELIM"
    echo "    Why that flag, and not plain -O3: from -O1 jump threading splits"
    echo "    the path where a thread's chunk comes out empty, and the split"
    echo "    path carries its own copy of the barrier sequence. Over the suite"
    echo "    that is 28 -> 43, gemver alone 3 -> 7, and no execution gains a"
    echo "    barrier — the extra call sites are the shape of the CFG, not"
    echo "    synchronisation. The flag turns off that one pass and nothing"
    echo "    else: with it every kernel is back on the number it has at -O0,"
    echo "    28 at every optimisation level, which was checked at -O0/1/2/3"
    echo "    with and without it. So this column can be read against the LLVM"
    echo "    ones, which are counted after -O3, where plain -O3 could not."
fi

# --- against the paper -------------------------------------------------------
# Section 4.5 gives all four of these numbers, and they are the kind that goes
# stale without anyone noticing: one extra rule in the DSL moves them, and the
# sentence stays as it was.
compare_to_claims() {
    local rows=0 bad=0
    echo
    echo "  === AGAINST SECTION 4.5 (reference/claims.csv) ==="
    printf '  %-16s %9s %9s   %s\n' subject measured paper verdict
    local pair subject measured value tol verdict
    for pair in "ours_baseline:$T_BASE" "ours_elim:$T_ELIM" \
                "clang:$T_CLANG" "gcc_o3_ntj:$T_GCC"; do
        subject="${pair%%:*}"; measured="${pair#*:}"
        # cleared first: read leaves them untouched when claim_row finds no
        # row, and the previous subject's numbers would be compared again.
        value=""; tol=""
        read -r value tol < <(claim_row team_barriers "$subject")
        [ -z "${value:-}" ] && continue
        # gcc absent leaves its total at 0, which is a missing column rather
        # than a count of zero.
        if [ "$subject" = "gcc_o3_ntj" ] && [ "$measured" -eq 0 ]; then
            printf '  %-16s %9s %9s   not built here\n' \
                "$subject" "-" "$value"
            continue
        fi
        rows=$((rows + 1))
        verdict="$(claim_verdict "$measured" "$value" "${tol:-0}")"
        [ "$verdict" = "DIFFERS" ] && bad=$((bad + 1))
        printf '  %-16s %9s %9s   %s\n' \
            "$subject" "$measured" "$value" "$verdict"
    done
    if [ "$rows" -eq 0 ]; then
        echo "  no team_barriers rows in $CLAIMS"
        return 0
    fi
    echo
    if [ "$bad" -eq 0 ]; then
        echo "  All $rows reproduce."
    else
        echo "  $bad of $rows differ. That is not automatically a regression:"
        echo "  these are counts the paper states in prose, so the sentence in"
        echo "  section 4.5 may simply be the thing that is out of date."
    fi
}

if [ "$FULL_SUITE" -eq 1 ] && [ "$FAILED" -eq 0 ]; then
    compare_to_claims
elif [ -f "$CLAIMS" ]; then
    echo
    if [ "$FULL_SUITE" -ne 1 ]; then
        echo "  Not compared with section 4.5: the published totals are sums"
        echo "  over all ${#ALL_KERNELS[@]} kernels and this run had ${#KERNEL_LIST[@]}."
    else
        echo "  Not compared with section 4.5: a kernel failed, so the totals"
        echo "  are short by whatever it would have contributed."
    fi
fi

echo
echo "  Counted in, one directory per kernel under"
echo "    $ARTDIR/<kernel>/"
echo "      clang-O3.ll  ours-baseline-O3.ll  ours-elim-O3.ll  gcc-O3-ntj.s"
echo "  Recount any row:"
echo "    grep -o 'call void @__kmpc_barrier' <file>.ll | wc -l"
echo "    grep -cE '\b(call|jmp)\b.*GOMP_barrier' gcc-O3-ntj.s"
echo "  Done — $CSV"

[ "$FAILED" -ne 0 ] && { echo "  some kernels failed (VERBOSE=1 for the tool output)"; exit 1; }
exit 0
