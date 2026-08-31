#!/bin/bash
# run_constructs.sh — end-to-end tests for the constructs and clauses PolyBench
# never writes. Across the whole suite it uses only `parallel`, a bare `for` and
# `private`, so everything else has no coverage without these.
#
# Each test is a standalone C program that prints 42 only if the clause did its
# job — not a diff against the reference compiler, which would agree and pass if
# both compilers ignored the clause. `nowait` and `proc_bind` cannot observe
# their own clause portably and say so in their headers; their evidence is in
# ../../Regression/.
#
# Usage:
#   ./run_constructs.sh                 # every test, $RUNTIME (default iomp)
#   RUNTIME=libgomp ./run_constructs.sh
#   ./run_constructs.sh num_threads     # one test, by name or path
#   FRONTEND=0 ./run_constructs.sh      # skip ClangIR, start from mlir/
#   KEEP=0 ./run_constructs.sh          # verdicts only, discard the IR
#   VERBOSE=1 ./run_constructs.sh       # let the tools say why one failed
#
# Every stage of the lowering is kept under results/<runtime>/constructs/<test>/;
# 02-lowered.mlir is the one to read. FRONTEND=0 is for a machine whose ClangIR
# is older than the clauses: the .c stops compiling before the lowering is
# reached, and mlir/ holds a copy of each module taken after the front-end.

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
# auto | 0 (pre-generated IR only) | 1 (front-end only) — see the header.
FRONTEND="${FRONTEND:-auto}"
case "$FRONTEND" in
    auto|0|1) ;;
    *) echo "ERROR: FRONTEND='$FRONTEND' is not auto, 0 or 1." >&2; exit 2 ;;
esac
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

frontend_to_mlir() {   # $1 = source, $2 = output .mlir, $3 = work dir
    local src="$1" out="$2" d="$3"
    "$CLANG" -S $CLANG_STRICT_FP $WARN_SUPPRESS \
        -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC_OMP" "$src" -o "$d/00-frontend.cir" 2>"$ERRSINK" || return 1
    "$CIR_OPT" "$d/00-frontend.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$out" 2>"$ERRSINK" || return 1
    # cir.* attributes survive the conversion and no dialect here owns them.
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$out"
}

mlir_to_bin() {   # $1 = input .mlir, $2 = output binary, $3 = work dir
    local in="$1" out="$2" d="$3"

    "$MLIR_OPT_OMP" --allow-unregistered-dialect \
        --omp-lower-dsl="$RULES" --omp-lower-runtime="$RUNTIME" \
        ${BARRIER_ELIM_FLAG:+$BARRIER_ELIM_FLAG} \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        "$in" > "$d/02-lowered.mlir" 2>"$ERRSINK" || return 1

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
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

for src in "${TESTS[@]}"; do
    name="$(basename "${src%.c}")"
    d="$OUTDIR/$name"
    [ "${KEEP:-1}" = "0" ] && d="$TMP/$name"
    rm -rf "$d"; mkdir -p "$d"

    if build_ref "$src" "$d/ref"; then
        ref_out="$("$d/ref" 2>/dev/null | head -1)"
    else
        ref_out="BUILD"
    fi

    # Which half of the pipeline this test got, and why. `via` ends up in the
    # verdict so a green run always says whether the front-end was exercised.
    twin="$SCRIPT_DIR/mlir/$name.mlir"
    ir="$d/01-cir-to-llvm.mlir"
    via=""
    case "$FRONTEND" in
        0)  cp "$twin" "$ir" 2>/dev/null && via="pre-generated IR" ;;
        1)  frontend_to_mlir "$src" "$ir" "$d" || via="FRONTEND" ;;
        *)  if frontend_to_mlir "$src" "$ir" "$d"; then
                via=""
            elif [ -f "$twin" ]; then
                cp "$twin" "$ir"; via="front-end refused, used pre-generated IR"
            else
                via="FRONTEND"
            fi ;;
    esac

    if [ "$via" = "FRONTEND" ]; then
        opt_out="FRONTEND"
    elif [ ! -f "$ir" ]; then
        opt_out="NO-IR"
    elif mlir_to_bin "$ir" "$d/opt" "$d"; then
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
    # Never let a fall-back pass silently: it proves the lowering, not the
    # front-end, and the two are different claims.
    if [ -n "$via" ] && [ "$via" != "FRONTEND" ]; then
        verdict="$verdict ${YELLOW}($via)${RESET}"
        csv="$csv;NO_FRONTEND"
    fi

    printf '  %-24s %-10s %-10s ' "$name" "$ref_out" "$opt_out"
    echo -e "$verdict"
    echo "$name;$ref_out;$opt_out;$csv" >> "$CSV"
done

echo
echo "  $PASS passed, $FAIL failed out of ${#TESTS[@]}"
if [ "${KEEP:-1}" != "0" ]; then
    echo
    echo "  Every stage is under $OUTDIR/<test>/ :"
    echo "    00-frontend.cir      what ClangIR emitted"
    echo "    01-cir-to-llvm.mlir  the omp + llvm dialects, our input"
    echo "    02-lowered.mlir      after mlir-opt-omp — the clause is now"
    echo "                         runtime calls, and nothing generic has run"
    echo "                         over them yet. Start here."
    echo "    03-llvm-dialect.mlir 04-llvmir.ll  05-opt-O3.ll  06-obj.o"
    echo
    echo "  To see what one clause costs, diff a lowering against its own"
    echo "  baseline — the rule file is the only thing that changed:"
    echo "    diff <(grep 'call @' $OUTDIR/parallel_if/02-lowered.mlir) \\"
    echo "         <(grep 'call @' $OUTDIR/num_threads/02-lowered.mlir)"
fi
echo "  Done — $CSV"

[ "$FAIL" -eq 0 ] || exit 1
exit 0
