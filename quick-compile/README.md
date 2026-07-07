# quick-compile

Quick test of the full pipeline(C → CIR → `mlir-opt-omp` → LLVM IR → object) on a tiny kernel.
Run the scripts from this directory.

- `test.c` — minimal OpenMP kernel.
- `main.c` — host driver that calls the kernel.

| Script | Runtime | Output |
|---|---|---|
| `compile-gomp.sh` | libgomp | `./test` + stock-compiler `./test-ref` — run both and diff by eye |
| `compile-iomp.sh` | iomp | same, with an extra MLIR/`opt -O3` optimization pipeline |
| `pulp/run.sh` | pmsis | riscv32 `test.o` → PULP-SDK link → run on gvsoc |

The pulp flow lives in [`pulp/`](pulp/), a minimal PULP-SDK app: `run.sh`
cross-compiles `../test.c` through `mlir-opt-omp` (pmsis rules) to a riscv32
`test.o` (needs `PULP_LLC` pointing at a riscv32-capable `llc`, +xpulpv),
then links it with `pulp_main.c` (fabric-controller boot), `cluster_main.c`
(the cluster-side counterpart of `main.c`) and `interface-adapter.c` (the
`ext_pi_*` shims the pmsis rules emit) and runs it on gvsoc. Without the GAP
SDK environment sourced it stops after producing `test.o`.

Prerequisites: clangir-enabled `clang`, `cir-opt`, `mlir-opt`,
`mlir-translate` on `PATH`, and `mlir-opt-omp` built in `../BUILD/`.
Intermediates (`*.cir`, `*.mlir`, `*.ll`, `*.o`) are left behind for
inspection; each script cleans them on the next run.

For whole-suite validation (PolyBench correctness/performance), see
[`test/Integration/`](../test/Integration/README.md).
