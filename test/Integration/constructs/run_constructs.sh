#!/bin/bash
# =============================================================================
# run_constructs.sh — end-to-end functional tests for the constructs and
# clauses PolyBench never writes.
#
# WHY THIS EXISTS. Across the whole 30-kernel PolyBench/OMP suite there are
# exactly three things: `#pragma omp parallel`, a bare `#pragma omp for`, and
# `private`. No firstprivate, no num_threads, no proc_bind, no nowait, no
# explicit schedule, no if, no barrier, no task, no taskwait. So of the
# construct/clause matrix the tool claims to support, the performance suite
# exercises four cells — and two of those only implicitly, through the default
# schedule of a bare `for` and the barrier that closes it. Everything else has
# no end-to-end coverage at all without these files.
#
# THE ORACLE. Every test is a standalone C program that prints one integer, and
# prints 42 only if the construct did its job. Not a comparison against a
# reference compiler: if a clause were silently ignored by BOTH compilers, a
# ref-vs-opt diff agrees and the test passes while nothing works. Each program
# instead observes the effect itself — the team really has the requested size,
# the copy really was taken at entry, every iteration really ran once — so a
# dropped clause prints something else, and prints what it saw rather than just
# failing.
#
# Each is run twice, once through the stock OpenMP compiler and once through
# the CIR / mlir-opt-omp pipeline. The reference run is not the oracle; it
# catches the other failure, a test that is simply wrong.
#
# TWO OF THEM CANNOT OBSERVE THEIR OWN CLAUSE, and say so in their headers
# rather than pretending: `nowait` is the absence of a barrier and `proc_bind`
# is a placement, neither of which a portable program can see without racing or
# measuring the machine. Their deterministic evidence is at IR level, in
# ../../Regression/. Here they assert the weaker but still useful property that
# the clause does not break the code around it.
#
# ADDING A TEST is adding a .c file to this directory. Nothing here enumerates
# them.
#
# Usage:
#   ./run_constructs.sh                 # every test, $RUNTIME (default iomp)
#   RUNTIME=libgomp ./run_constructs.sh
#   ./run_constructs.sh num_threads     # one test, by name or path
#   KEEP=1 ./run_constructs.sh          # keep the per-stage IR for inspection
#   VERBOSE=1 ./run_constructs.sh       # let the tools say why one failed
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Nothing here is a PolyBench kernel and nothing is cross-compiled: this driver
# only needs the host toolchain, and must not require the PULP SDK.
SKIP_PULP_SDK=1
# shellcheck source=../lib/common.sh
. "$SCRIPT_DIR/../lib/common.sh"

if [ "$TARGET" != "native" ]; then
    echo "ERROR: RUNTIME=$RUNTIME is not a host runtime." >&2
    echo "       These are plain C programs linked against libomp or libgomp;" >&2
    echo "       there is nothing to run them on under pmsis." >&2
    exit 2
fi

EXPECTED=42
OUTDIR="${OUTDIR:-$PWD/results}/$RUNTIME/constructs"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_constructs.csv"

ERRSINK="/dev/null"
[ -n "${VERBOSE:-}" ] && ERRSINK="/dev/stderr"

# --- The two pipelines -------------------------------------------------------
# ref: the stock OpenMP compiler for this runtime, straight through.
build_ref() {   # $1 = source, $2 = output binary
    "$REF_CC" -O2 $REF_FP $WARN_SUPPRESS -fopenmp $REF_OMP_INC \
        "$1" -o "$2" 2>"$ERRSINK"
}

# opt: C -> ClangIR -> mlir-opt-omp -> LLVM IR -> object -> link.
# Each stage is written out under $d so a failure can be read rather than
# guessed at; KEEP=1 leaves them behind.
build_opt() {   # $1 = source, $2 = output binary, $3 = work dir
    local src="$1" out="$2" d="$3"
    mkdir -p "$d"

    "$CLANG" -S $CLANG_STRICT_FP $WARN_SUPPRESS \
        -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC_OMP" "$src" -o "$d/00-frontend.cir" 2>"$ERRSINK" || return 1
    "$CIR_OPT" "$d/00-frontend.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$d/01-cir-to-llvm.mlir" 2>"$ERRSINK" || return 1
    # cir.* attributes survive the conversion and no dialect here owns them.
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$d/01-cir-to-llvm.mlir"

    "$MLIR_OPT_OMP" --allow-unregistered-dialect \
        --omp-lower-dsl="$RULES" --omp-lower-runtime="$RUNTIME" \
        ${BARRIER_ELIM_FLAG:+$BARRIER_ELIM_FLAG} \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        "$d/01-cir-to-llvm.mlir" > "$d/02-lowered.mlir" 2>"$ERRSINK" || return 1

    "$MLIR_OPT" "$d/02-lowered.mlir" \
        --canonicalize --cse --sccp --symbol-dce \
        --loop-invariant-code-motion --canonicalize --cse \
        --convert-arith-to-llvm --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$d/03-llvm-dialect.mlir" 2>"$ERRSINK" || return 1
    "$MLIR_TRANSLATE" "$d/03-llvm-dialect.mlir" --mlir-to-llvmir \
        > "$d/04-llvmir.ll" 2>"$ERRSINK" || return 1
    "$OPT" -S -O3 "$d/04-llvmir.ll" > "$d/05-opt-O3.ll" 2>"$ERRSINK" || return 1
    "$LLC" -O3 -relocation-model=pic -filetype=obj "$d/05-opt-O3.ll" \
        -o "$d/06-obj.o" 2>"$ERRSINK" || return 1
    # $OPT_FOPENMP / $OPT_EXTRA_LIBS select the runtime library to link.
    "$CLANG" $OPT_FOPENMP -no-pie "$d/06-obj.o" $OPT_EXTRA_LIBS -o "$out" \
        2>"$ERRSINK" || return 1
}

# --- Which tests -------------------------------------------------------------
# A name, a path, or nothing at all for every .c in this directory.
if [ $# -gt 0 ]; then
    TESTS=()
    for arg in "$@"; do
        if [ -f "$arg" ]; then TESTS+=("$arg")
        elif [ -f "$SCRIPT_DIR/$arg" ]; then TESTS+=("$SCRIPT_DIR/$arg")
        elif [ -f "$SCRIPT_DIR/$arg.c" ]; then TESTS+=("$SCRIPT_DIR/$arg.c")
        else echo "ERROR: no such test: $arg" >&2; exit 2
        fi
    done
else
    mapfile -t TESTS < <(find "$SCRIPT_DIR" -maxdepth 1 -name '*.c' | sort)
fi

if [ "${#TESTS[@]}" -eq 0 ]; then
    echo "ERROR: no .c tests found in $SCRIPT_DIR" >&2
    exit 2
fi

echo "=== CONSTRUCTS & CLAUSES — end-to-end functional tests ==="
echo "runtime: $RUNTIME    ref: $REF_CC    expected output: $EXPECTED"
echo "tests:   ${#TESTS[@]}"
echo

printf '  %-24s %-10s %-10s %s\n' test ref opt verdict
echo "kernel;ref;opt;verdict" > "$CSV"

PASS=0; FAIL=0
TMP="$(mktemp -d)"
cleanup() { [ -n "${KEEP:-}" ] || rm -rf "$TMP"; }
trap cleanup EXIT

for src in "${TESTS[@]}"; do
    name="$(basename "${src%.c}")"
    d="$TMP/$name"
    [ -n "${KEEP:-}" ] && d="$OUTDIR/$name"
    mkdir -p "$d"

    if build_ref "$src" "$d/ref"; then
        ref_out="$("$d/ref" 2>/dev/null | head -1)"
    else
        ref_out="BUILD"
    fi

    if build_opt "$src" "$d/opt" "$d"; then
        opt_out="$("$d/opt" 2>/dev/null | head -1)"
    else
        opt_out="BUILD"
    fi

    # The oracle is the value, not the agreement: a clause both compilers drop
    # would agree at something that is not 42.
    if [ "$opt_out" = "$EXPECTED" ]; then
        verdict="${GREEN}${BOLD}PASS${RESET}"; csv=PASS; PASS=$((PASS + 1))
    else
        verdict="${RED}${BOLD}FAIL${RESET}"; csv=FAIL; FAIL=$((FAIL + 1))
    fi
    # A reference that does not print 42 means the test itself is wrong, which
    # is a different bug and must not be reported as a lowering failure.
    if [ "$ref_out" != "$EXPECTED" ]; then
        verdict="$verdict ${YELLOW}(ref says $ref_out — the test is suspect)${RESET}"
        csv="$csv;REF_SUSPECT"
    fi

    printf '  %-24s %-10s %-10s ' "$name" "$ref_out" "$opt_out"
    echo -e "$verdict"
    echo "$name;$ref_out;$opt_out;$csv" >> "$CSV"
done

echo
echo "  $PASS passed, $FAIL failed out of ${#TESTS[@]}"
[ -n "${KEEP:-}" ] && echo "  per-stage IR kept under $OUTDIR/<test>/"
echo "  Done — $CSV"

[ "$FAIL" -eq 0 ] || exit 1
exit 0
