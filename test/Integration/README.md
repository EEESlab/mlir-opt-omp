# Integration tests — end-to-end correctness & performance

Two drivers share the same setup (`lib/common.sh`, which pulls in the kernel
lists from `lib/kernels.sh`, the host pipelines from `lib/native.sh` and — for
`pmsis` — the PULP target from `lib/pulp.sh`), against the same PolyBench
kernels and the same `run.env`:

- **`run_correctness.sh`** — compiles each kernel twice and checks the two runs
  produce **bit-identical** array dumps.
- **`run_performance.sh`** — times our tool against the native compiler and
  reports speedups (see [Performance](#performance) below).
- **`run_barrier_stats.sh`** — counts the team barriers `--omp-barrier-elim`
  removes from each kernel (see [Barrier elimination](#barrier-elimination)).

A third, lighter driver covers the task construct (it lives in
[`tasks/`](tasks/) together with its test cases):

- **`tasks/run_tasks.sh`** — end-to-end smoke test for `omp.task` (libgomp or
  iomp, selected by the first argument or `RUNTIME`), two checks, both run
  against the real runtime library and expecting `42`:
  - **[1] MLIR** — a hand-written `parallel { task { *p = 42 } }` module
    ([`tasks/task_nested.mlir`](tasks/task_nested.mlir)) lowered through
    `mlir-opt-omp` + the MLIR/LLVM tools and linked against the runtime
    (libgomp or libomp). Starts from MLIR, so it does not need the CIR
    front-end to emit `omp.task`.
  - **[2] C** — [`tasks/task_smoke.c`](tasks/task_smoke.c) compiled both with
    the stock OpenMP compiler (ref: gcc for libgomp, clang for iomp) and
    through the full CIR / `mlir-opt-omp` pipeline (opt); outputs must match.
    This path depends on ClangIR emitting `omp.task`; if your `clang-cir`
    lacks task support, [2] fails at the front-end while [1] still passes.

  Run: `tasks/run_tasks.sh [libgomp|iomp]` (results land in
  `results/<runtime>/tasks/` regardless of the working directory).

In both, the two compilers are:

- **ref** — a stock OpenMP compiler (clang for `iomp`, gcc for `libgomp`, the
  PULP-SDK gcc for `pmsis`)
- **opt** — the full CIR/MLIR pipeline through `mlir-opt-omp`

Three runtimes are supported by the same drivers and the same `run.env`:
`iomp` and `libgomp` compile and run on the host; `pmsis` cross-compiles for
PULP/GAP8 and runs on the **gvsoc** simulator (see
[PULP / gvsoc](#pulp--gvsoc-runtimepmsis) below).

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

Nothing is hard-coded to a machine. Configuration is split in two, by how often
it changes:

| File | Holds | Required? |
|---|---|---|
| [`local.env`](../../local.env.example) (repo root) | where the tools are | yes — and shared with `quick-compile/` |
| `run.env` (here) | what the tests run | no, everything has a default |

So the setup is one file:

```sh
cd ../..                                  # repo root
cp local.env.example local.env
$EDITOR local.env        # LLVM_BIN, OMP_TOOL_BIN, INC_OMP
```

Those three are enough for the bundled kernels: your ClangIR build, your
`mlir-opt-omp` build, and the OpenMP headers of your GCC. The drivers check all
three at startup and tell you which one is wrong rather than failing later
inside a compile.

Add `run.env` only when you want different defaults from run to run:

```sh
cp run.env.example run.env         # in this directory
```

Precedence is `local.env` → `run.env` → whatever you set inline, so
`RUNTIME=libgomp ./run_correctness.sh` always wins. Both files are git-ignored.
Setups for machines the project is developed on are kept in
[`docs/setups/`](../../docs/setups/); copy one of those instead if it matches
yours.

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

Everything lands under `$OUTDIR/<runtime>` (default `./results/<runtime>/`),
one folder per runtime — `iomp/`, `libgomp/`, `pmsis/` — so runs against
different runtimes never overwrite each other:

```
results/
  iomp/                           # (same layout under libgomp/ and pmsis/)
    results_correctness.csv       # kernel;PASS|FAIL|ERROR
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

Each cell is run `REPS` times (default 10); the min and max are dropped and the
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
PLOT=true SUITE=full ./run_performance.sh              # + speedup chart
```

### Speedup chart

Set `PLOT=true` (run.env or inline) to render a bar chart of the
**self-relative parallel speedup** per kernel once the run finishes — native
(`ref_seq/ref_par`) vs our tool (`opt_seq/opt_par`), i.e. the `speedup_native`
and `speedup_opt` columns. It covers whatever ran (`bundled`, `full`, or an
explicit `KERNELS` list) and lands at `results/<runtime>/results_performance_<sel>.png`,
where `<sel>` names the selection — the suite (`full`/`bundled`) or, for an
explicit `KERNELS` list, the kernel basename(s) — plus the dataset size, e.g.
`_full_large` or `_gemm-omp_mini`. The
native bar is labelled by runtime — *Clang frontend* (`iomp`), *GCC frontend*
(`libgomp`) or *PULP-SDK GCC* (`pmsis`).

The rendering is done by [`lib/plot_speedup.py`](lib/plot_speedup.py) and needs
`python3` + `matplotlib`/`numpy`; if they are missing the run still succeeds
and only the plot is skipped. The recommended setup is a local venv (auto-picked
when present; git-ignored):

```sh
python3 -m venv .venv
.venv/bin/pip install matplotlib numpy
```

Python resolution order: `PLOT_PYTHON` (if set), then `./.venv/bin/python`,
then `python3` from PATH. You can also run the script by hand on any existing
CSV, e.g. for a vector figure:

```sh
python3 lib/plot_speedup.py results/libgomp/results_performance.csv fig.pdf --runtime libgomp
```

> The perf script defaults to `DATASET=LARGE_DATASET` (correctness defaults to
> `MINI`). At `MINI` the parallel run is pure thread-spawn overhead and the
> speedups are meaningless — keep it `LARGE`/`EXTRALARGE` for real numbers.

Output:

```
results/
  <runtime>/                         # iomp/, libgomp/ or pmsis/
    results_performance.csv          # per-kernel rows + a GEOMEAN summary row
    results_performance_<sel>.png    # speedup chart (when PLOT=true); <sel> =
                                     # suite (full/bundled) or kernel name(s),
                                     # + dataset size (e.g. _full_large)
    <kernel>-omp/performance/        # the four binaries, their .ll, and *.log timings
```

## Barrier elimination

`--omp-barrier-elim` drops team barriers the surrounding OpenMP structure
already guarantees — chiefly the implicit barrier of a work-sharing loop that
ends a parallel region, which the team join makes redundant. It runs on the
`omp` dialect before any runtime is chosen, so the same removals apply to all
three; what differs is only what a barrier costs.

Two things to measure, and they answer different questions.

`run_barrier_stats.sh` counts what disappears, without running anything:

```sh
SUITE=full ./run_barrier_stats.sh        # -> results/results_barrier_stats.csv
```

The count is **static** — barriers removed from the program text, not barrier
executions saved. A loop whose barrier sits inside a sequential outer loop
counts once here and fires once per iteration at run time, so the dynamic
saving is the larger number. `floyd-warshall` is the clearest case: one static
barrier, executed once per `k`.

`BARRIER_ELIM=1` puts the pass in the real pipeline, so the other two drivers
measure its effect. Run each configuration twice, changing only this variable:

```sh
BARRIER_ELIM=0 SUITE=full ./run_performance.sh    # baseline
BARRIER_ELIM=1 SUITE=full ./run_performance.sh    # with the optimisation
BARRIER_ELIM=1 SUITE=full ./run_correctness.sh    # still bit-identical?
```

Both drivers write into the same `results/<runtime>/` tree, so move or rename
the baseline CSV before the second run or it will be overwritten.

## PULP / gvsoc (`RUNTIME=pmsis`)

The same two drivers also target PULP/GAP8 through the **gvsoc** simulator.
This only runs on machines with the GAP SDK + gvsoc installed. Every `PULP_*`
variable is listed under
[Configuration reference](#pulp--gvsoc-runtimepmsis-1) and belongs in
`local.env`. Two working setups are kept under [`docs/setups/`](../../docs/setups/):
`lucap-workstation.env` — one setup for all three runtimes, switched via
`RUNTIME=...` at launch — and `pulp-tagliavini.env` for tagliavini's machine
(copy one to `local.env`).

Instead of compiling host binaries, each cell is one invocation of the
PolyBench-PULP harness Makefile (`PULP_APP_DIR`), which builds **and** runs on
gvsoc in one shot:

|              | sequential                  | parallel                            |
|--------------|-----------------------------|-------------------------------------|
| native (ref) | `make ... KERNEL_SRC=k.c`   | `make ... OMP_NATIVE=1` (SDK OpenMP)|
| our tool(opt)| `kernel.o` (no omp) + `make ... OMP_OPT=1` | `kernel.o` (omp) + `make ... OMP_OPT=1` |

For the opt cells, `lib/pulp.sh` (sourced by `lib/common.sh` when `RUNTIME=pmsis`)
cross-compiles the kernel to
`$PULP_APP_DIR/kernel.o` first: clang→CIR → `mlir-opt-omp`
(`--omp-lower-runtime=pmsis`) → LLVM IR → `$PULP_LLC` (riscv32, `+xpulpv`).
`PULP_OPT`/`PULP_LLC` point at a RISC-V-capable LLVM install, which may differ
from the host tools.

The pmsis rules emit calls to a small `ext_pi_*` shim layer over the PMSIS
API (fork/barrier/core-id); the harness provides and links those shims. A
reference implementation is kept at
[`quick-compile/pulp/interface-adapter.c`](../../quick-compile/pulp/interface-adapter.c).

Differences from the native runtimes:

- **Cycles** are scraped from the `Cycles = N` line the harness prints on the
  gvsoc console; each cell runs **once** (`REPS` ignored — the simulator is
  deterministic). `THREADS` is ignored too (core count is fixed by the harness
  `NUM_CORES`).
- **Correctness** compares the `==BEGIN/END DUMP_ARRAYS==` sections extracted
  from the two gvsoc console logs (ref = `OMP_NATIVE=1`, opt = `OMP_OPT=1`).
- **Performance** appends four `size_*` columns (bytes of the linked ELF,
  `$PULP_BUILD_BIN`) to `results_performance.csv`; the speedup/geomean columns
  are unchanged.
- `DATASET` defaults to `MINI_DATASET` (GAP8 memory) and — like
  `PULP_POLYBENCH_DEFS` — **must match what the harness Makefile defines** for
  the native builds, otherwise ref and opt are not comparable.

```sh
RUNTIME=pmsis ./run_correctness.sh                 # needs run.env for pulp
RUNTIME=pmsis ./run_performance.sh
RUNTIME=pmsis PULP_VERBOSE=1 ./run_performance.sh linear-algebra/blas/gemm/gemm-omp.c
```

Per-cell build/run logs are kept under `results/pmsis/<kernel>-omp/...` (`run.log`,
`ref_seq.log`, ...), together with the final `.ll` of the opt kernels.

## Configuration reference

Every variable read by the drivers, with its default. **Tools**, **Paths** and
the `PULP_*` locations belong in `local.env` at the repo root; **Run
parameters**, **Performance** and **Advanced** belong in `run.env` here. The
split is a convention, not a constraint — the two files are sourced in order, so
either can hold anything. You can also export any of them in your shell or pass
them inline, which wins over both files.

### Tools

`LLVM_BIN` and `OMP_TOOL_BIN` are prepended to `PATH`, so the tools below
resolve to the right build. Both default to empty (everything taken from
`PATH`); if one is set but does not hold the expected binary, the drivers print
a warning and fall back to `PATH`.

| Variable       | Default | Meaning                                        |
|----------------|---------|------------------------------------------------|
| `LLVM_BIN`     | *empty* | dir of clang/opt/llc/cir-opt/mlir-\*           |
| `OMP_TOOL_BIN` | *empty* | dir of `mlir-opt-omp`                          |
| `CLANG`        | `clang` | override an individual tool by name or path…   |
| `GCC`          | `gcc`   | (ref compiler for `libgomp`)                   |
| `OPT` / `LLC`  | `opt` / `llc` |                                          |
| `CIR_OPT`      | `cir-opt` |                                              |
| `MLIR_OPT` / `MLIR_TRANSLATE` | `mlir-opt` / `mlir-translate` |         |
| `MLIR_OPT_OMP` | `mlir-opt-omp` |                                         |

### Paths

| Variable         | Default | Meaning                                      |
|------------------|---------|----------------------------------------------|
| `INC_OMP`        | `/usr/lib/gcc/x86_64-linux-gnu/13/include` | OpenMP headers for the clang→CIR front-end; must match the local GCC |
| `POLYBENCH`      | `kernels/` (vendored) | PolyBench-OpenMP checkout root, required for `SUITE=full` |
| `POLYBENCH_UTIL` | `$POLYBENCH/utilities` | PolyBench support headers           |
| `RULES`          | the repo's `rules.dsl` | DSL file passed to `mlir-opt-omp`   |
| `OUTDIR`         | `$PWD/results` | binaries, dumps and CSVs land in `$OUTDIR/<runtime>` |

### Run parameters

| Variable  | Default | Meaning                                             |
|-----------|---------|-----------------------------------------------------|
| `RUNTIME` | `iomp`  | `iomp`, `libgomp` or `pmsis` (PULP/gvsoc)           |
| `SUITE`   | `bundled` | `bundled` (vendored kernels) or `full`            |
| `KERNELS` | *empty* | explicit space-separated kernel list; overrides `SUITE`, paths resolved against `$POLYBENCH` |
| `DATASET` | `MINI_DATASET` for correctness, `LARGE_DATASET` for performance, always `MINI_DATASET` on `pmsis` unless set | PolyBench dataset size macro |
| `THREADS` | `16`    | thread count for the parallel runs; ignored on `pmsis` |
| `BARRIER_ELIM` | `0` | `1` adds `--omp-barrier-elim` to the pipeline. Off by default so a plain run is the baseline to compare against |

### Performance only (`run_performance.sh`)

| Variable            | Default | Meaning                                   |
|---------------------|---------|-------------------------------------------|
| `REPS`              | `10`     | timed runs per cell, min+max dropped; ignored on `pmsis` (gvsoc is deterministic) |
| `VARIANCE_ACCEPTED` | `5`     | warn if a cell's relative std-dev exceeds this % |
| `PLOT`              | `false` | `true` → render `results/<runtime>/results_performance_<sel>.png` (needs matplotlib) |
| `PLOT_PYTHON`       | *empty* | python to render with; otherwise `./.venv`, then `python3` |

### PULP / gvsoc (`RUNTIME=pmsis`)

See [PULP / gvsoc](#pulp--gvsoc-runtimepmsis) for what these drive.

| Variable         | Default | Meaning                                      |
|------------------|---------|----------------------------------------------|
| `PULP_APP_DIR`   | *none, required* | PolyBench-PULP harness dir (the PULP-SDK Makefile) |
| `PULP_TOOLCHAIN_BIN` | *empty* | GAP RISC-V GCC toolchain bin dir (prepended to `PATH`) |
| `PULP_SDK_ENV`   | *empty* | GAP SDK env script, sourced automatically when set |
| `PULP_OPT` / `PULP_LLC` | host `$OPT` / `$LLC` | riscv32-capable LLVM `opt`/`llc` (xpulpv), often a different install |
| `PULP_OPT_FLAGS` | *empty* | extra host `opt` pass on the `.ll`           |
| `PULP_LLC_FLAGS` | `-O3 -mtriple=riscv32-unknown-elf -mattr=+m,+c,+xpulpv -relocation-model=pic` | |
| `PULP_PLATFORM`  | `gvsoc` | `gvsoc` or `board`                           |
| `PULP_MAKE` / `PULP_MAKE_ARGS` | `make` / *empty* | harness make binary and extra args |
| `PULP_BUILD_BIN` | `BUILD/GAP8_V3/GCC_RISCV_PULPOS/test` | linked ELF relative to `PULP_APP_DIR`, used for the size columns |
| `PULP_POLYBENCH_DEFS` | `-DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME` | defines baked into the opt `kernel.o`; **must match** the harness Makefile's native builds |
| `PULP_VERBOSE`   | `0`     | `1` → stream the make/gvsoc output instead of logging it |

### Advanced

| Variable          | Default | Meaning                                     |
|-------------------|---------|---------------------------------------------|
| `CLANG_STRICT_FP` | `-ffp-contract=off -fno-vectorize -fno-slp-vectorize` | see below |
| `GCC_STRICT_FP`   | `-ffp-contract=off -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize` | see below |
| `WARN_SUPPRESS`   | `-Wno-ignored-attributes` | silences the harmless warnings clang emits parsing GCC's `omp.h` |
| `POLYBENCH_LFLAGS`| *empty* | extra link flags; running as root auto-adds `-DPOLYBENCH_LINUX_FIFO_SCHEDULER` and `-lc` |
| `OMP_PLACES`      | `cores` | exported for the parallel runs              |
| `OMP_PROC_BIND`   | `true`  | exported for the parallel runs              |

The strict FP flags are enabled by default and must match between ref and opt —
without them FMA contraction and reordered reductions make iterative kernels
diverge even though both binaries are IEEE-correct. Override only if you know
why.
