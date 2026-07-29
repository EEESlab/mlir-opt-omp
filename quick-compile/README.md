# quick-compile

The place to start if you want to see the tool work on something small. Two
entry points, depending on how much of the pipeline you care about — run the
scripts from this directory.

**From MLIR** (`compile-from-mlir.sh`) — just the three passes plus the standard
LLVM tail. The input is already what `mlir-opt-omp` consumes, so this needs
**no ClangIR and no `cir-opt`**: LLVM/MLIR 23  and a built
`mlir-opt-omp` are enough. It ends by checking the output itself, so a bare
`./compile-from-mlir.sh` is a yes/no answer on whether your build lowers
correctly.

**From C** (`compile-gomp.sh`, `compile-iomp.sh`, `pulp/run.sh`) — the full
C → CIR → `mlir-opt-omp` → LLVM IR → object pipeline, which is what this
project is really built around. Needs a clangir-enabled `clang` and `cir-opt`.

The two meet in the middle: `test.mlir` is the hand-written twin of `test.c`
(same `@add` symbol, same `main.c`, same expected numbers), and the
`test-s1.mlir` the C scripts leave behind is a valid input to
`compile-from-mlir.sh` — feeding it back in runs the second half of the C pipeline
on its own.

- `test.c` — minimal OpenMP kernel: `#pragma omp parallel for` over `c[i] = a[i] + b[i]`.
- `test.mlir` — the same kernel written directly in the `omp` + `llvm` dialects.
- `main.c` — host driver that calls the kernel; prints 11…110.

| Script | Front-end | Runtime | Output |
|---|---|---|---|
| `compile-from-mlir.sh [rt] [in.mlir]` | none — starts from MLIR | libgomp (default) or iomp | `./test`, run and checked against the expected numbers |
| `compile-gomp.sh` | ClangIR | libgomp | `./test` + stock-compiler `./test-ref` — run both and diff by eye |
| `compile-iomp.sh` | ClangIR | iomp | same, with an extra MLIR/`opt -O3` optimization pipeline |
| `pulp/run.sh` | ClangIR | pmsis | riscv32 `test.o` → PULP-SDK link → run on gvsoc |

The pulp flow lives in [`pulp/`](pulp/), a minimal PULP-SDK app: `run.sh`
cross-compiles `../test.c` through `mlir-opt-omp` (pmsis rules) to a riscv32
`test.o` (needs `PULP_LLC` pointing at a riscv32-capable `llc`, +xpulpv), then
links it with `pulp_main.c` (fabric-controller boot), `cluster_main.c`
(the cluster-side counterpart of `main.c`) and `interface-adapter.c` (the
`ext_pi_*` shims the pmsis rules emit) and runs it on gvsoc. Without the GAP
SDK environment sourced it stops after producing `test.o`.

Prerequisites: `mlir-opt`, `mlir-translate`, `opt`, `llc`, a built
`mlir-opt-omp` and a compiler to link with — plus, for the C scripts only, a
clangir-enabled `clang` and `cir-opt`. The scripts locate them through
[`local.env`](../local.env.example) at the repo root — the same file the
Integration tests use — and report at startup anything that is missing or
points at the wrong place. Without a `local.env` they fall back to whatever is
on `PATH`, with `mlir-opt-omp` taken from `../build/`.

For the pulp flow, `local.env` also supplies `PULP_LLC` and `PULP_SDK_ENV`,
which `run.sh` sources for you.

Intermediates are left behind for inspection and cleaned on the next run:
`test-s*` for the C scripts (`s1` = post-`cir-opt`, the point where
`mlir-opt-omp` takes over), `test-m*` for `compile-from-mlir.sh`.

For whole-suite validation (PolyBench correctness/performance), see
[`test/Integration/`](../test/Integration/README.md).
