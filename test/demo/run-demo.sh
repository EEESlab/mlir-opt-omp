#!/usr/bin/env bash
# =============================================================================
# run-demo.sh — the full lowering pipeline for the slides, STARTING FROM C and
# INCLUDING the ClangIR front end.
#
# One kernel (vecadd.c: `#pragma omp parallel for`) is taken all the way down,
# and every stage is written to out/ so each slide can show one step:
#
#   front end (runtime-independent):
#     vecadd.c
#       --(clang -fclangir -emit-cir)-------> out/00-frontend.cir       [ClangIR / CIR dialect]
#       --(cir-opt --cir-to-llvm)-----------> out/01-cir-to-llvm.mlir   [omp + llvm dialects]
#                                             ^ this is the input to mlir-opt-omp
#   per runtime (iomp | libgomp | pmsis):
#       --(mlir-opt-omp --omp-to-omp-lower)-> out/<rt>/02-omp-to-omp-lower.mlir  [omp_lower.construct]
#       --(       + --omp-outline)----------> out/<rt>/03-omp-outline.mlir       [outlined func + packed captures]
#       --(       + --omp-lower-plan)-------> out/<rt>/04-omp-lower-plan.mlir     [__kmpc_* / GOMP_* / ext_pi_*]
#       --(mlir-opt|translate|opt|llc|clang)-> out/<rt>/vecadd  (native) --> run
#         (pmsis stops at 04: the riscv32 / gvsoc back end needs the PULP SDK.)
#
# Tool locations are read from test/Integration/config.env — the SAME file the
# Integration drivers use. Copy an example once:
#     cp ../Integration/config.env.lucap-wsl.example ../Integration/config.env
# or point the vars inline:
#     LLVM_BIN=... OMP_TOOL_BIN=... INC_OMP=... ./run-demo.sh
#
# Usage:
#     ./run-demo.sh [runtime ...]        # default: iomp libgomp pmsis
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEG="$HERE/../Integration"

# --- config: reuse the Integration config.env for tool paths ----------------
if [[ -f "$INTEG/config.env" ]]; then set -a; . "$INTEG/config.env"; set +a; fi
[[ -n "${LLVM_BIN:-}" ]]     && PATH="$LLVM_BIN:$PATH"
[[ -n "${OMP_TOOL_BIN:-}" ]] && PATH="$OMP_TOOL_BIN:$PATH"
export PATH

CLANG="${CLANG:-clang}"
CIR_OPT="${CIR_OPT:-cir-opt}"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
OPT="${OPT:-opt}"
LLC="${LLC:-llc}"
MLIR_OPT_OMP="${MLIR_OPT_OMP:-mlir-opt-omp}"
RULES="${RULES:-$HERE/../../rules.dsl}"
# OpenMP headers for the clang->CIR front end; match your GCC version.
INC_OMP="${INC_OMP:-/usr/lib/gcc/x86_64-linux-gnu/13/include}"

OUT="$HERE/out"
RUNTIMES=("$@"); [[ ${#RUNTIMES[@]} -eq 0 ]] && RUNTIMES=(iomp libgomp pmsis)

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: '$1' not found. Set the paths in $INTEG/config.env" >&2
    echo "       (cp ../Integration/config.env.lucap-wsl.example ../Integration/config.env)" >&2
    exit 1
  }
}
need "$CLANG"; need "$CIR_OPT"; need "$MLIR_OPT_OMP"

rm -rf "$OUT"; mkdir -p "$OUT"
echo "clang        : $(command -v "$CLANG")"
echo "cir-opt      : $(command -v "$CIR_OPT")"
echo "mlir-opt-omp : $(command -v "$MLIR_OPT_OMP")"
echo "rules.dsl    : $RULES"
echo

# --- front end (runtime-independent): C -> ClangIR -> omp+llvm MLIR ----------
echo "== front end: vecadd.c -> ClangIR -> MLIR =="
"$CLANG" -S -Xclang -fclangir -Xclang -emit-cir -fopenmp -Wno-ignored-attributes \
    -I"$INC_OMP" "$HERE/vecadd.c" -o "$OUT/00-frontend.cir"
"$CIR_OPT" "$OUT/00-frontend.cir" --cir-to-llvm --reconcile-unrealized-casts \
    -o "$OUT/01-cir-to-llvm.mlir"
# Strip leftover cir.* attributes the downstream tools do not register.
sed -i -E 's/cir\.[^,}]+,? ?//g' "$OUT/01-cir-to-llvm.mlir"
echo "   -> out/00-frontend.cir , out/01-cir-to-llvm.mlir"

# Driver: stock clang, compiled once (only the kernel needs the CIR path).
"$CLANG" -O3 -c "$HERE/driver.c" -o "$OUT/driver.o"
echo

omp_flags=(--allow-unregistered-dialect --omp-lower-dsl="$RULES")

for rt in "${RUNTIMES[@]}"; do
  echo "== runtime: $rt =="
  d="$OUT/$rt"; mkdir -p "$d"
  in="$OUT/01-cir-to-llvm.mlir"

  # The three mlir-opt-omp passes, run incrementally so each stage is captured.
  "$MLIR_OPT_OMP" "${omp_flags[@]}" --omp-lower-runtime="$rt" \
      --omp-to-omp-lower                             "$in" -o "$d/02-omp-to-omp-lower.mlir"
  "$MLIR_OPT_OMP" "${omp_flags[@]}" --omp-lower-runtime="$rt" \
      --omp-to-omp-lower --omp-outline               "$in" -o "$d/03-omp-outline.mlir"
  "$MLIR_OPT_OMP" "${omp_flags[@]}" --omp-lower-runtime="$rt" \
      --omp-to-omp-lower --omp-outline --omp-lower-plan "$in" -o "$d/04-omp-lower-plan.mlir"
  echo "   IR: out/$rt/02-omp-to-omp-lower.mlir .. 04-omp-lower-plan.mlir"

  # Native back end + run (iomp / libgomp). pmsis is IR-only here.
  case "$rt" in
    iomp)    fopenmp="-fopenmp";         libs="-lm";;
    libgomp) fopenmp="-fopenmp=libgomp"; libs="-lm -lgomp";;
    *)  echo "   (pmsis: IR only here — to build & RUN on gvsoc use pulp/run-pulp.sh)"; echo; continue;;
  esac

  "$MLIR_OPT" "$d/04-omp-lower-plan.mlir" \
      --canonicalize --cse --sccp --symbol-dce --loop-invariant-code-motion \
      --canonicalize --cse --convert-arith-to-llvm --convert-func-to-llvm \
      --reconcile-unrealized-casts -o "$d/05-llvm-dialect.mlir"
  "$MLIR_TRANSLATE" "$d/05-llvm-dialect.mlir" --mlir-to-llvmir > "$d/06-llvmir.ll"
  "$OPT" -S -O3 "$d/06-llvmir.ll" -o "$d/07-opt-O3.ll"
  "$LLC" -O3 -relocation-model=pic -filetype=obj "$d/07-opt-O3.ll" -o "$d/08-kernel.o"
  "$CLANG" $fopenmp -no-pie "$d/08-kernel.o" "$OUT/driver.o" $libs -o "$d/vecadd"

  echo -n "   run ($rt, OMP_NUM_THREADS=4): "
  OMP_NUM_THREADS=4 "$d/vecadd" | tr '\n' ' '; echo "  (expected: 0 11 22 33 44 55 66 77)"
  echo
done

echo "All stages under $OUT/"
echo "  00/01 = shared front end (ClangIR) ; 02-04 = per-runtime lowering"
