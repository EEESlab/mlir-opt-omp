#!/bin/bash
# run_barrier_vs_native.sh — counts team-barrier call sites: ours with and
# without --omp-barrier-elim, against clang's and gcc's on the same kernels.
#
# Not a runtime comparison to configure: each side is counted in the form it
# emits. The three LLVM columns are counted in LLVM IR after -O3 (__kmpc_barrier),
# gcc in its assembly at -O0 (GOMP_barrier), because from -O1 jump threading
# and friends duplicate paths and each copy carries the barrier call site.
#
# Usage:
#   ./run_barrier_vs_native.sh                       # kernels in the suite
#   ./run_barrier_vs_native.sh path/to/kernel-omp.c
#   VERBOSE=1 ./run_barrier_vs_native.sh             # let the tools report why
#
# Leaves results_barrier_vs_native.csv.
#
# On a full-suite run the four totals are printed next to what section 4.5
# states, from reference/claims.csv. Side by side and nothing more: what the
# difference means is the reader's call.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Nothing is executed and nothing is built for the target.
SKIP_PULP_SDK=1

# Fixed, and not something to select: our side is counted against the iomp ABI
# because that is what the LLVM columns speak, gcc against its own.
RUNTIME=iomp
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

# No dumps and no timer: nothing here is run, and both sides must see the same
# defines or they would not be compiling the same program.
POLYBENCH_CFLAGS="$POLYBENCH_ROOT_CFLAGS"

OUTDIR="${OUTDIR:-$PWD/results}/barrier_vs_native"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_barrier_vs_native.csv"

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

# -O0: gcc decides the elision in the front end, so it is already applied
# there, and no later pass has duplicated a barrier call site yet. From -O1
# jump threading and the other passes that duplicate a path to remove a
# branch copy the barrier's block with it, so the count rises without any
# execution gaining a barrier (the suite goes 28 -> 43 at -O3, gemver 3 -> 7).
build_gcc() {   # $1 = source, $2 = output .s
    "$GCC" -fopenmp -O0 -S $GCC_STRICT_FP \
        -I"$INC" -I"$(dirname "$1")" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$1" -o "$2" 2>"$ERRSINK"
}

# From -O2 gcc emits the last barrier of a region as a tail call, which prints
# as jmp: anchoring on `call` alone would drop those. It does not arise at -O0,
# but the pattern costs nothing and keeps a count at another stage honest.
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
echo "dataset:   $DATASET"
echo "polybench: $POLYBENCH"
echo "kernels:   ${#KERNEL_LIST[@]}"
echo "files:     $OUTDIR"
echo "form:      combined = '#pragma omp parallel for'"
echo "           split    = '#pragma omp parallel' + '#pragma omp for'"
echo

echo "kernel,pragma_form,clang,gcc_o0,ours_baseline,ours_barrier_elim" > "$CSV"
printf '  %-22s %-9s %6s %6s %8s %13s\n' \
    kernel form clang gcc base barrier_elim

# Intermediates only (.cir and the MLIR stages); the counted files go straight
# to $OUTDIR and stay there.
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

FAILED=0
T_CLANG=0; T_GCC=0; T_BASE=0; T_ELIM=0

for k in "${KERNEL_LIST[@]}"; do
    src="$(resolve_src "$k")" || { echo "  SKIP (not found): $k"; continue; }
    name="$(basename "$src" .c)"
    form="$(pragma_form "$src")"
    kdir="$OUTDIR/$name"
    mkdir -p "$kdir"

    "$CLANG" -fopenmp -O3 -S -emit-llvm $CLANG_STRICT_FP $WARN_SUPPRESS \
        -I"$INC" -I"$(dirname "$src")" -I"$INC_OMP" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" -o "$kdir/clang-O3.ll" 2>"$ERRSINK" \
        || { echo "  ERROR (clang): $name"; echo "$name,$form,,,," >> "$CSV"; FAILED=1; continue; }

    build_ours "$src" "$kdir/ours-baseline-O3.ll" "" \
        || { echo "  ERROR (ours, baseline): $name"; echo "$name,$form,,,," >> "$CSV"; FAILED=1; continue; }
    build_ours "$src" "$kdir/ours-elim-O3.ll" "--omp-barrier-elim" \
        || { echo "  ERROR (ours, barrier-elim): $name"; echo "$name,$form,,,," >> "$CSV"; FAILED=1; continue; }

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

    T_CLANG=$((T_CLANG + c)); T_BASE=$((T_BASE + b)); T_ELIM=$((T_ELIM + e))
    printf '  %-22s %-9s %6s %6s %8s %13s\n' \
        "$name" "$form" "$c" "${g:--}" "$b" "$e"
    echo "$name,$form,$c,$g,$b,$e" >> "$CSV"
done

echo
TOTALS="  clang: $T_CLANG ; ours without the pass: $T_BASE ; ours with the pass: $T_ELIM"
# gcc missing leaves its total at 0: drop the item rather than print a zero.
[ "$T_GCC" -gt 0 ] && TOTALS="$TOTALS ; gcc: $T_GCC"
echo "$TOTALS"

if [ "$T_GCC" -gt 0 ]; then
    echo "    Counted at -O0: gcc already elides the barrier in its front end."
    echo "    From -O1 the passes that duplicate a path to remove a branch"
    echo "    copy the block the barrier sits in along with it: more call"
    echo "    sites, no execution gaining a barrier."
fi

# --- against the paper -------------------------------------------------------
# Section 4.5 gives all four of these numbers, and they are the kind that goes
# stale without anyone noticing: one extra rule in the DSL moves them, and the
# sentence stays as it was.
compare_to_claims() {
    local rows=0
    echo
    echo "  === SECTION 4.5 (reference/claims.csv) ==="
    printf '  %-16s %9s %9s\n' subject measured paper
    local pair subject measured value tol
    for pair in "ours_baseline:$T_BASE" "ours_elim:$T_ELIM" \
                "clang:$T_CLANG" "gcc_o0:$T_GCC"; do
        subject="${pair%%:*}"; measured="${pair#*:}"
        # cleared first: read leaves them untouched when claim_row finds no
        # row, and the previous subject's numbers would be compared again.
        value=""; tol=""
        read -r value tol < <(claim_row team_barriers "$subject")
        [ -z "${value:-}" ] && continue
        # gcc absent leaves its total at 0, which is a missing column rather
        # than a count of zero.
        if [ "$subject" = "gcc_o0" ] && [ "$measured" -eq 0 ]; then
            measured="-"          # gcc absent: a missing column, not a zero
        fi
        rows=$((rows + 1))
        printf '  %-16s %9s %9s\n' "$subject" "$measured" "$value"
    done
    if [ "$rows" -eq 0 ]; then
        echo "  no team_barriers rows in $CLAIMS"
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
echo "    $OUTDIR/<kernel>/"
echo "      clang-O3.ll  ours-baseline-O3.ll  ours-elim-O3.ll  gcc-O0.s"
echo "  Recount any row:"
echo "    grep -o 'call void @__kmpc_barrier' <file>.ll | wc -l"
echo "    grep -cE '\b(call|jmp)\b.*GOMP_barrier' gcc-O0.s"
echo "  Done — $CSV"

[ "$FAILED" -ne 0 ] && { echo "  some kernels failed (VERBOSE=1 for the tool output)"; exit 1; }
exit 0
