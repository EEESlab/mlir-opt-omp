# Integration tests — end-to-end correctness

`run_correctness.sh` compiles each PolyBench kernel twice and checks that the
two runs produce **bit-identical** array dumps:

- **ref** — a stock OpenMP compiler (clang for `iomp`, gcc for `libgomp`)
- **opt** — the full CIR/MLIR pipeline through `mlir-opt-omp`

This is the slow, whole-pipeline layer (needs clang-cir, cir-opt, mlir tools,
an OpenMP runtime and a PolyBench checkout). For fast per-pass IR checks see
`../Regression/` (lit + FileCheck).

## Bundled kernels

A couple of PolyBench kernels are vendored under [`kernels/`](kernels) so the
test runs **self-contained**, without an external checkout:

- `linear-algebra/blas/gemm/` — dense matrix multiply
- `linear-algebra/kernels/atax/` — matrix-vector product
- `utilities/` — `polybench.c` / `polybench.h` support code

By default `POLYBENCH` points here and only these kernels run (`SUITE=bundled`).

## Setup

Nothing is hard-coded to a machine. For the bundled kernels you only need the
toolchain reachable. Either export the variables in your shell, or — easier —
copy the example config and edit it:

```sh
cp config.env.example config.env
$EDITOR config.env        # set LLVM_BIN, OMP_TOOL_BIN, ...
```

`config.env` is git-ignored, so your local paths stay out of the repo.

## Run

```sh
./run_correctness.sh                          # bundled kernels, defaults
RUNTIME=libgomp ./run_correctness.sh          # switch runtime
DATASET=SMALL_DATASET THREADS=8 ./run_correctness.sh
./run_correctness.sh linear-algebra/blas/gemm/gemm-omp.c   # single kernel
```

Run the **full** PolyBench suite against an external checkout:

```sh
SUITE=full POLYBENCH=/path/to/PolyBenchC-4.2.1-OpenMP ./run_correctness.sh
```

Or pass an explicit list:

```sh
KERNELS="linear-algebra/blas/gemm/gemm-omp.c stencils/adi/adi-omp.c" \
  POLYBENCH=/path/to/checkout ./run_correctness.sh
```

A kernel path is resolved relative to `$POLYBENCH` (or taken as-is if it already
points to a file).

## Output

Everything lands under `$OUTDIR` (default `./results/`):

```
results/
  results_correctness.csv         # kernel;PASS|FAIL|ERROR
  <kernel>-omp/ref/{<bin>,dump.txt}
  <kernel>-omp/opt/{<bin>,<bin>.ll,dump.txt}   # final LLVM IR kept for debugging
```

The script exits non-zero if any kernel is not PASS, so it can gate CI.

## Configuration reference

All variables, with their defaults, are documented in
[`config.env.example`](config.env.example). The most important:

| Variable       | Meaning                                             |
|----------------|-----------------------------------------------------|
| `LLVM_BIN`     | dir of clang/opt/llc/cir-opt/mlir-* (prepended to PATH) |
| `OMP_TOOL_BIN` | dir of `mlir-opt-omp` (prepended to PATH)           |
| `POLYBENCH`    | PolyBench-OpenMP checkout root                       |
| `RULES`        | DSL file (defaults to the repo's `rules.dsl`)        |
| `INC_OMP`      | OpenMP headers for the clang→CIR front-end           |
| `RUNTIME`      | `iomp` or `libgomp`                                  |
| `DATASET`      | PolyBench dataset size macro                         |
| `THREADS`      | `OMP_NUM_THREADS`                                    |
| `SUITE`        | `bundled` (vendored kernels) or `full`              |
| `KERNELS`      | explicit space-separated kernel list (overrides `SUITE`) |

Strict FP flags (`-ffp-contract=off`, no auto-vectorisation) are enabled by
default and must match between ref and opt — without them FMA contraction and
reordered reductions make iterative kernels diverge even though both binaries
are IEEE-correct.
