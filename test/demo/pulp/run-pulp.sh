#!/usr/bin/env bash
# =============================================================================
# run-pulp.sh — the pmsis variant of the demo that actually RUNS on gvsoc.
#
# The SAME kernel as the host demo (../vecadd.c: `#pragma omp parallel for`),
# taken all the way to a GAP8 executable and run on the gvsoc simulator:
#
#   ../vecadd.c
#     --(clang -fclangir -emit-cir)------> kernel.cir                 ClangIR
#     --(cir-opt --cir-to-llvm)----------> kernel-01-cir-to-llvm.mlir omp + llvm dialects
#     --(mlir-opt-omp runtime=pmsis)-----> kernel-04-omp-lower-plan.mlir  ext_pi_cl_team_fork / _barrier
#     --(mlir-opt | mlir-translate)------> kernel-06.ll               LLVM IR
#     --(PULP_LLC riscv32 +xpulpv)-------> kernel.o                   GAP8 object
#     --(PULP-SDK make + gvsoc)----------> RUN: prints 0 11 22 33 44 55 66 77
#
# The FC boot, cluster dispatch and ext_pi_* shims live next to this script
# (pulp_main.c, cluster_main.c, interface-adapter.c, Makefile) — a trimmed copy
# of quick-compile/pulp — so the demo is self-contained: no external PolyBench
# harness needed, unlike test/Integration's pmsis path.
#
# Config comes from ../../Integration/config.env (the workstation example has it
# all): LLVM_BIN, OMP_TOOL_BIN, RULES, INC_OMP, and the PULP knobs PULP_LLC,
# PULP_SDK_ENV, PULP_TOOLCHAIN_BIN.
#     cp ../../Integration/config.env.lucap-workstation.example ../../Integration/config.env
#     ./run-pulp.sh
#
# Runs only on a machine with the GAP SDK + gvsoc installed (the workstation).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEG="$HERE/../../Integration"

# --- config: reuse the Integration config.env for tool + PULP paths ----------
if [[ -f "$INTEG/config.env" ]]; then set -a; . "$INTEG/config.env"; set +a; fi

# GAP SDK env: sets RULES_DIR, puts gvsoc + toolchain on PATH. SDK scripts
# reference unset vars, so relax `set -u` while sourcing — same as lib/pulp.sh.
[[ -n "${PULP_TOOLCHAIN_BIN:-}" ]] && PATH="$PULP_TOOLCHAIN_BIN:$PATH"
if [[ -n "${PULP_SDK_ENV:-}" ]]; then
  set +u
  # shellcheck disable=SC1090
  . "$PULP_SDK_ENV" || { echo "error: could not source PULP_SDK_ENV=$PULP_SDK_ENV" >&2; exit 2; }
  set -u
fi
# Put our LLVM/CIR tools + mlir-opt-omp back in front so they beat the SDK's.
[[ -n "${LLVM_BIN:-}" ]]     && PATH="$LLVM_BIN:$PATH"
[[ -n "${OMP_TOOL_BIN:-}" ]] && PATH="$OMP_TOOL_BIN:$PATH"
export PATH

CLANG="${CLANG:-clang}"
CIR_OPT="${CIR_OPT:-cir-opt}"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
MLIR_OPT_OMP="${MLIR_OPT_OMP:-mlir-opt-omp}"
RULES="${RULES:-$HERE/../../../rules.dsl}"
INC_OMP="${INC_OMP:-/usr/lib/gcc/x86_64-linux-gnu/12/include}"
PULP_LLC="${PULP_LLC:?set PULP_LLC to a riscv32-capable llc (+xpulpv) in config.env}"
PULP_LLC_FLAGS="${PULP_LLC_FLAGS:--O3 -mtriple=riscv32-unknown-elf -mattr=+m,+c,+xpulpv -relocation-model=pic}"
PULP_PLATFORM="${PULP_PLATFORM:-gvsoc}"

KERNEL="$HERE/../vecadd.c"

need() { command -v "$1" >/dev/null 2>&1 || { echo "error: '$1' not found — check config.env" >&2; exit 1; }; }
need "$CLANG"; need "$CIR_OPT"; need "$MLIR_OPT_OMP"; need "$PULP_LLC"

echo "clang        : $(command -v "$CLANG")"
echo "mlir-opt-omp : $(command -v "$MLIR_OPT_OMP")"
echo "PULP_LLC     : $PULP_LLC"
echo "rules.dsl    : $RULES"
echo

cd "$HERE"
rm -f kernel.cir kernel-*.mlir kernel-*.ll kernel.o

set -e
echo "== ../vecadd.c -> ClangIR -> MLIR(omp) -> pmsis lowering -> riscv32 kernel.o =="
"$CLANG" -S -Xclang -fclangir -Xclang -emit-cir -fopenmp -Wno-ignored-attributes \
    -I"$INC_OMP" "$KERNEL" -o kernel.cir
"$CIR_OPT" kernel.cir --cir-to-llvm --reconcile-unrealized-casts \
    -o kernel-01-cir-to-llvm.mlir
sed -i -E 's/cir\.[^,}]+,? ?//g' kernel-01-cir-to-llvm.mlir
"$MLIR_OPT_OMP" --allow-unregistered-dialect --omp-lower-dsl="$RULES" \
    --omp-lower-runtime=pmsis --omp-to-omp-lower --omp-outline --omp-lower-plan \
    kernel-01-cir-to-llvm.mlir > kernel-04-omp-lower-plan.mlir
"$MLIR_OPT" kernel-04-omp-lower-plan.mlir \
    --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
    -o kernel-05-llvm-dialect.mlir
"$MLIR_TRANSLATE" kernel-05-llvm-dialect.mlir --mlir-to-llvmir > kernel-06.ll
"$PULP_LLC" $PULP_LLC_FLAGS -filetype=obj kernel-06.ll -o kernel.o
echo "   -> kernel.o (GAP8 riscv32, +xpulpv)"
echo

# The GAP SDK env must be sourced for the link+run step (RULES_DIR / gvsoc).
if [[ -z "${RULES_DIR:-}" ]]; then
  echo "kernel.o built, but RULES_DIR is unset — the GAP SDK env was not sourced."
  echo "Set PULP_SDK_ENV in config.env (or source the SDK yourself), then re-run"
  echo "to link and run on gvsoc."
  exit 0
fi

echo "== link (PULP-SDK) + run on $PULP_PLATFORM =="
echo "   expected console output: 0 11 22 33 44 55 66 77"
make clean all run "platform=$PULP_PLATFORM"
