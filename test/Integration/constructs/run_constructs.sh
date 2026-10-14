#!/bin/bash
# run_constructs.sh — MLIR-only end-to-end tests for constructs and clauses.
#
# Usage:
#   ./run_constructs.sh                 # every test, $RUNTIME (default iomp)
#   RUNTIME=libgomp ./run_constructs.sh
#   ./run_constructs.sh num_threads     # one test, by name or path (.mlir)
#   KEEP=0 ./run_constructs.sh          # verdicts only, discard the intermediate IR
#   VERBOSE=1 ./run_constructs.sh       # let the tools say why one failed

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_PULP_SDK=1
# shellcheck source=../lib/common.sh
. "$SCRIPT_DIR/../lib/common.sh"

if [ "$TARGET" != "native" ]; then
    echo "ERROR: RUNTIME=$RUNTIME is not a host runtime." >&2
    exit 2
fi

EXPECTED=42
OUTDIR="${OUTDIR:-$PWD/results}/$RUNTIME/constructs"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results_constructs.csv"

ERRSINK="/dev/null"
[ -n "${VERBOSE:-}" ] && ERRSINK="/dev/stderr"

# --- MLIR Lowering Pipeline ---------------------------------------------------
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
    "$CLANG" $OPT_FOPENMP -no-pie "$d/06-obj.o" $OPT_EXTRA_LIBS -o "$out" \
        2>"$ERRSINK" || return 1
}

# --- Target Test Resolution --------------------------------------------------
if [ $# -gt 0 ]; then
    TESTS=()
    for arg in "$@"; do
        if [ -f "$arg" ]; then TESTS+=("$arg")
        elif [ -f "$SCRIPT_DIR/$arg" ]; then TESTS+=("$SCRIPT_DIR/$arg")
        elif [ -f "$SCRIPT_DIR/$arg.mlir" ]; then TESTS+=("$SCRIPT_DIR/$arg.mlir")
        else echo "ERROR: no such test: $arg" >&2; exit 2
        fi
    done
else
    mapfile -t TESTS < <(find "$SCRIPT_DIR" -maxdepth 1 -name '*.mlir' | sort)
fi

if [ "${#TESTS[@]}" -eq 0 ]; then
    echo "ERROR: no .mlir tests found in $SCRIPT_DIR" >&2
    exit 2
fi

echo "=== CONSTRUCTS & CLAUSES — MLIR lowering functional tests ==="
echo "runtime: $RUNTIME    expected output: $EXPECTED"
echo "tests:   ${#TESTS[@]}"
echo

printf '  %-30s %-10s %s\n' test opt verdict
echo "kernel;opt;verdict" > "$CSV"

PASS=0; FAIL=0
TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

for src in "${TESTS[@]}"; do
    name="$(basename "${src%.mlir}")"
    d="$OUTDIR/$name"
    [ "${KEEP:-1}" = "0" ] && d="$TMP/$name"
    rm -rf "$d"; mkdir -p "$d"

    if mlir_to_bin "$src" "$d/opt" "$d"; then
        opt_out="$("$d/opt" 2>/dev/null | head -1)"
    else
        opt_out="BUILD"
    fi

    if [ "$opt_out" = "$EXPECTED" ]; then
        verdict="${GREEN}${BOLD}PASS${RESET}"; csv=PASS; PASS=$((PASS + 1))
    else
        verdict="${RED}${BOLD}FAIL${RESET}"; csv=FAIL; FAIL=$((FAIL + 1))
    fi

    printf '  %-30s %-10s ' "$name" "$opt_out"
    echo -e "$verdict"
    echo "$name;$opt_out;$csv" >> "$CSV"
done

echo
echo "  $PASS passed, $FAIL failed out of ${#TESTS[@]}"
if [ "${KEEP:-1}" != "0" ]; then
    echo
    echo "  Every stage is under $OUTDIR/<test>/ :"
    echo "    02-lowered.mlir      after mlir-opt-omp"
    echo "    03-llvm-dialect.mlir 04-llvmir.ll  05-opt-O3.ll  06-obj.o"
fi
echo "  Done — $CSV"

[ "$FAIL" -eq 0 ] || exit 1
exit 0