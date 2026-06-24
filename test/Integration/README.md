# Integration tests — end-to-end correctness & performance

Two drivers share one compile pipeline (`common.sh`), against the same
PolyBench kernels and the same `config.env`:

- **`run_correctness.sh`** — compiles each kernel twice and checks the two runs
  produce **bit-identical** array dumps.
- **`run_performance.sh`** — times our tool against the native compiler and
  reports speedups (see [Performance](#performance) below).

A third, lighter driver covers the task construct:

- **`run_tasks.sh`** — end-to-end smoke test for `omp.task` (libgomp). It lowers
  a hand-written `parallel { task { *p = 42 } }` MLIR module
  ([`tasks/task_nested.mlir`](tasks/task_nested.mlir)) through `mlir-opt-omp` +
  the MLIR/LLVM tools, links `-lgomp`, runs it, and asserts the output is `42`.
  It starts from MLIR (not C), so it does not need the CIR front-end to emit
  `omp.task`. Run: `./run_tasks.sh`.

In both, the two compilers are:

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

## Performance

`run_performance.sh` builds a 2×2 matrix per kernel and times each cell with
PolyBench's cycle-accurate TSC timer:

|              | sequential (1T)        | parallel (`$THREADS`)        |
|--------------|------------------------|------------------------------|
| native (ref) | `ref_seq` (`-O3`)      | `ref_par` (`-O3 -fopenmp`)   |
| our tool(opt)| `opt_seq` (no omp)     | `opt_par` (CIR/MLIR -fopenmp)|

Each cell is run `REPS` times (default 5); the min and max are dropped and the
mean ± std-dev of the rest is reported. A cell whose relative std-dev exceeds
`VARIANCE_ACCEPTED`% (default 5) is flagged as noisy.

The sequential cells are compiled **without** `-fopenmp` (the `#pragma omp` are
ignored → serial code). PolyBench kernels that call `omp_get_thread_num()` /
`omp_get_num_threads()` unconditionally would then fail to link, so the seq
builds pull in `omp_stubs.c` — serial OpenMP stubs (one thread, id 0). No
OpenMP runtime is attached to the sequential baseline.

Reported ratios:

| Metric              | Formula             | Meaning                              |
|---------------------|---------------------|--------------------------------------|
| `speedup_native`    | `ref_seq / ref_par` | native self seq→par speedup          |
| `speedup_opt`       | `opt_seq / opt_par` | our self seq→par speedup             |
| `opt_vs_native_par` | `ref_par / opt_par` | **headline**: our parallel vs native (>1 = we win) |
| `opt_vs_native_seq` | `ref_seq / opt_seq` | same, sequential                     |

The suite summary uses the **geometric mean** of each ratio across all kernels
(the standard way to average benchmark speedups).

```sh
./run_performance.sh                                   # bundled kernels
RUNTIME=libgomp DATASET=LARGE_DATASET THREADS=16 ./run_performance.sh
./run_performance.sh linear-algebra/blas/gemm/gemm-omp.c   # single kernel
SUITE=full POLYBENCH=/path/to/checkout ./run_performance.sh
```

> The perf script defaults to `DATASET=LARGE_DATASET` (correctness defaults to
> `MINI`). At `MINI` the parallel run is pure thread-spawn overhead and the
> speedups are meaningless — keep it `LARGE`/`EXTRALARGE` for real numbers.

Output:

```
results/
  results_performance.csv         # per-kernel rows + a GEOMEAN summary row
  <kernel>-omp/performance/        # the four binaries, their .ll, and *.log timings
```

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
| `THREADS`      | thread count for parallel runs                       |
| `SUITE`        | `bundled` (vendored kernels) or `full`              |
| `KERNELS`      | explicit space-separated kernel list (overrides `SUITE`) |
| `REPS`         | (perf) timed runs per cell — min+max dropped         |
| `VARIANCE_ACCEPTED` | (perf) warn if a cell's relative std-dev exceeds this % |

Strict FP flags (`-ffp-contract=off`, no auto-vectorisation) are enabled by
default and must match between ref and opt — without them FMA contraction and
reordered reductions make iterative kernels diverge even though both binaries
are IEEE-correct.
