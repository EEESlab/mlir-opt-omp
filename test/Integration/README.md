# Integration tests — end-to-end correctness & performance

Two drivers share the same setup (`lib/common.sh`, which pulls in the kernel
lists from `lib/kernels.sh`, the host pipelines from `lib/native.sh` and — for
`pmsis` — the PULP target from `lib/pulp.sh`), against the same PolyBench
kernels and the same `run.env`:

- **`run_correctness.sh`** — compiles each kernel twice and checks the two runs
  produce **bit-identical** array dumps.
- **`run_performance.sh`** — times our tool against the native compiler and
  reports speedups (see [Performance](#performance) below).
- **`run_barrier_vs_native.sh`** — counts the team barriers left with and
  without `--omp-barrier-elim`, against clang's and gcc's count on the same
  kernels (see [Barrier elimination](#barrier-elimination)).

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

## Setup

Nothing is hard-coded to a machine. Configuration is split in two, by how often
it changes:

| File | Holds | Required? |
|---|---|---|
| [`local.env`](../../local.env.example) (repo root) | where the tools are | yes |
| `run.env` (here) | what the tests run | no, everything has a default |

So the setup is one file:

```sh
cd ../..                                  # repo root
cp local.env.example local.env
$EDITOR local.env        # LLVM_BIN, OMP_TOOL_BIN, INC_OMP
```

Four values: your ClangIR build, your `mlir-opt-omp` build, the OpenMP headers
of your GCC, and `POLYBENCH` — a checkout of PolyBench/OMP, since this repo
vendors no kernels of its own. The drivers check all four at startup and say
which one is wrong rather than failing later inside a compile.

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
./run_correctness.sh                          # whole suite, defaults
RUNTIME=libgomp ./run_correctness.sh          # switch runtime
DATASET=SMALL_DATASET THREADS=8 ./run_correctness.sh
./run_correctness.sh linear-algebra/blas/gemm/gemm-omp.c   # single kernel
```

`POLYBENCH` normally lives in `local.env`, but can be given per run:

```sh
POLYBENCH=/path/to/PolyBenchC-4.2.1-OpenMP ./run_correctness.sh
```

Or pass an explicit list of kernels:

```sh
KERNELS="linear-algebra/blas/gemm/gemm-omp.c stencils/adi/adi-omp.c" \
  POLYBENCH=/path/to/checkout ./run_correctness.sh
```

A kernel path is resolved relative to `$POLYBENCH` (or taken as-is if it already
points to a file).

## Output

Everything lands under `$OUTDIR/<runtime>` (default `./results/<runtime>/`),
one folder per runtime — `iomp/`, `libgomp/`, `pmsis/` — so runs against
different runtimes never overwrite each other. `BARRIER_ELIM=1` appends
`-barrier-elim` to that folder (`results/iomp-barrier-elim/`), keeping an
optimised run beside the baseline it is compared with:

```
results/
  iomp/                           # (same layout under libgomp/ and pmsis/)
    results_correctness.csv       # kernel;PASS|FAIL|ERROR
    <kernel>-omp/ref/{<bin>,dump.txt}
    <kernel>-omp/opt/{<bin>,<bin>.ll,dump.txt}   # final LLVM IR kept for debugging
```

The script exits non-zero if any kernel is not PASS, so it can gate CI.

## Performance

### Reproducing a figure

One config per figure, in [`configs/`](configs/). `RUN_ENV` reads one **instead
of** `run.env`, so a stale personal `run.env` cannot contaminate the run and the
command line says which configuration produced the numbers:

```sh
RUN_ENV=configs/paper-libgomp.env PLOT=true ./run_performance.sh   # Figure 4
RUN_ENV=configs/paper-iomp.env    PLOT=true ./run_performance.sh   # Figure 5
RUN_ENV=configs/paper-pmsis.env   PLOT=true ./run_performance.sh   # Figures 6 and 7
```

Each is hours: 30 kernels × 4 cells × 10 repetitions at `LARGE`. For a shorter
run put `REPS=` on the command line, where an environment variable wins over the
file — `REPS=5` halves it and keeps a standard deviation over three samples,
`REPS=3` is 3.3× faster and reports the median of three with no deviation at
all. The **dataset is deliberately not the knob**: fewer repetitions cost
confidence in a value, a smaller dataset changes which value is being measured
and the result stops being comparable with the figure. On `pmsis` neither
applies — gvsoc is deterministic, so every cell runs once and `DATASET` is
already pinned to `MINI` by the GAP8 memory budget.

A name that matches no file is a hard error listing what exists, rather than a
silent fall-back to the defaults.

### Reading the result against the paper

`run_performance.sh` ends by printing this itself; `COMPARE=false` turns it off,
and it can be re-run on any CSV already on disk:

```sh
python3 lib/compare_to_reference.py results/libgomp/results_performance.csv \
  --runtime libgomp
```

The four checks are ordered by how well they survive a change of machine, which
is the whole difficulty: a reviewer runs on a different CPU, so an absolute
speedup is not comparable at all.

| # | check | compares against | transfers? |
|---|---|---|---|
| 1 | **parity** — `speedup_opt / speedup_native` per kernel | the run itself | **yes** — a property of the compiler, and the paper's central claim |
| 2 | **named kernels** — `doitgen` ahead on libgomp, `floyd-warshall`/`deriche`/`nussinov` behind on pmsis | §4.2 and §4.3, which name them | yes, they are claims about a mechanism |
| 3 | **size** increase below 0.7% (pmsis only) | §4.3, an exact number | yes, decided by the compiler |
| 4 | **absolute** speedups | [`reference/`](reference/) | **no** — orientation only |

Check 1 needs no reference file: it is computed from the CSV that has just been
written, and it is the strongest of the four because `preserves performance
across all benchmarks` is a statement about the two bars, not about their
height. Check 4 is printed as a single summary line rather than per kernel,
because its reference values were read off the published charts by eye *and*
measured on other hardware — a difference there is two kinds of noise before it
is ever a finding.

Check 1 also prints the absolute `opt_vs_native` figures beside the parity one,
because the driver's own summary table shows them and the two look like they
disagree — a backend that emits slower code reads ~0.89 there while parity
reads ~1.00. Whether that is a contradiction is settled by comparing the
parallel figure with the sequential one: if they match, the deficit is uniform,
which makes it code quality rather than parallelisation and is precisely why it
cancels in the self-relative ratio.

A named kernel that stops reproducing is **not** treated as a failure — it
usually means the sentence in the paper has aged, which is worth knowing before
submission. Only check 3 can exit non-zero, and only under `--strict`; the rest
are readings, not assertions.

A section that does not apply to the runtime says so rather than vanishing, so
the numbering never has a hole in it.

For the reference values themselves, and the numbers the paper states exactly,
see [`reference/`](reference/).

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
./run_performance.sh                                   # whole suite
RUNTIME=libgomp DATASET=LARGE_DATASET THREADS=16 ./run_performance.sh
./run_performance.sh linear-algebra/blas/gemm/gemm-omp.c   # single kernel
POLYBENCH=/path/to/checkout ./run_performance.sh
PLOT=true ./run_performance.sh              # + speedup chart
```

### Speedup chart

Set `PLOT=true` (run.env or inline) to render a bar chart of the
**self-relative parallel speedup** per kernel once the run finishes — native
(`ref_seq/ref_par`) vs our tool (`opt_seq/opt_par`), i.e. the `speedup_native`
and `speedup_opt` columns. It covers whatever ran (the whole suite, or an
explicit `KERNELS` list) and lands at `results/<runtime>/results_performance_<sel>.png`,
where `<sel>` names the selection — `suite` for the whole set or, for an
explicit `KERNELS` list, the kernel basename(s) — plus the dataset size, e.g.
`_suite_large` or `_gemm-omp_mini`. The
native bar is labelled by runtime — *Clang frontend* (`iomp`), *GCC frontend*
(`libgomp`) or *PULP-SDK GCC* (`pmsis`).

On `RUNTIME=pmsis` a **second chart** is rendered beside it,
`..._size.png`: the binary size change of each parallel build against its own
sequential one, from the `size_*` columns. It is signed — a toolchain whose
parallel build comes out smaller plots below zero — and it exists only on the
PULP path, where the footprint is a result rather than a footnote. By hand on
a CSV already on disk:

```sh
python3 lib/plot_speedup.py results/pmsis/results_performance.csv fig.pdf \
  --metric size --runtime pmsis
```

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
                                     # "suite" or the kernel name(s),
                                     # + dataset size (e.g. _suite_large)
    <kernel>-omp/performance/        # the four binaries, their .ll, and *.log timings
```

Under `BARRIER_ELIM=1` those two names take a `_barrier-elim` suffix as well —
`results_performance_barrier-elim.csv`,
`results_performance_suite_large_barrier-elim.png`. The folder already says
which run they belong to, but these are the files that leave it, and a figure
in a paper directory has no folder left to tell it by.

## Barrier elimination

`--omp-barrier-elim` drops team barriers the surrounding OpenMP structure
already guarantees — chiefly the implicit barrier of a work-sharing loop that
ends a parallel region, which the team join makes redundant. It runs on the
`omp` dialect before any runtime is chosen, so the same removals apply to all
three; what differs is only what a barrier costs.

Two things to measure, and they answer different questions: how many barrier
call sites disappear, and what that is worth at run time.

The statistics half is runtime-independent, the emitted half is not, so run it
once per runtime you care about. Either way the count is **static** — barriers
removed from the program text, not barrier executions saved.

`run_barrier_vs_native.sh` asks the other question — how many barriers are left
compared with the compiler people actually use:

```sh
./run_barrier_vs_native.sh   # -> results/iomp/results_barrier_vs_native.csv
```

Both sides are counted in LLVM IR after `-O3`, so neither is measured at a
kinder stage than the other. On PolyBench the totals are clang 45, our pipeline
59 without the pass and 26 with it: **19 fewer than clang, 42%**. The dataset
does not enter into it — `MINI` and `LARGE` give the same table row for row, on
every column including gcc's, since the macros move loop bounds and the count is
static. The
`pragma_form` column explains the whole delta. clang elides a work-sharing
loop's trailing barrier only for the *combined* `#pragma omp parallel for`; on
the split `parallel { ... for ... }` it emits it, and there our baseline
matches clang kernel for kernel while the pass removes one per region. The pass
reasons about structure on the `omp` dialect rather than about which directive
was written, so it covers both spellings.

The three LLVM columns are iomp, so they speak one ABI. **gcc gets a column of
its own, counted before `-O3`** (`gcc -fopenmp -O0 -S`, counting `GOMP_barrier`
call sites), because from `-O1` on gcc spreads the same barriers over more call
sites than the program needs: over this suite 28 at `-O0` becomes 43 at `-O3`,
gemver alone 3 becomes 7. That number would measure duplication as much as
synchronisation. gcc decides the elision in the front-end, so it is already
applied at `-O0`; our own count does not move between stages — the same 59/25
in MLIR and after `-O3` — so the comparison holds, but say which stage when you
quote it.

That stage choice was checked, not assumed. Counting gcc's barriers at `-O0`,
`-O1`, `-O2` and `-O3`, then again at each level with `-fno-thread-jumps`, puts
every one of the 30 kernels back on its `-O0` number exactly: 28 at every level.
So one pass accounts for the whole 28 → 43 — jump threading splits the path
where a thread's chunk comes out empty, and the split path carries its own copy
of the barrier sequence. No execution gains a barrier. That makes `-O0` not the
convenient stage but the only one that measures the elision instead of the CFG
shape.

It is *not* loop cloning: `GCC_STRICT_FP` already turns the vectoriser off, so
no kernel gets a vector and a scalar copy. Four `omp for` in one region are
enough to see the effect on its own — 3 call sites at `-O0`, 6 from `-O1`, 3
again with `-fno-thread-jumps`.

Count `GOMP_barrier`, not `call GOMP_barrier`: from `-O2` gcc emits the last
barrier of a region as a tail call, which prints as `jmp` and which an anchor
on `call` silently drops — 6 of the 43 at `-O3`, and none at `-O0`.

What that column says is worth knowing before quoting the clang one: **gcc
performs this elision too**, and lands at 28 where the pass lands at 26, kernel
for kernel on 28 of 30. The two compilers miss it in different places — clang
on the `split` spelling, gcc on a region that holds a declaration, which puts
the loop inside a block and out of reach of its check (add one line to
`gemm-omp.c` and gcc's count goes 0 → 1). One rule on the dialect covers both,
and covers a third runtime whose toolchain has no such optimisation at all.

A loop whose barrier sits inside a sequential outer loop
counts once here and fires once per iteration at run time, so the dynamic
saving is the larger number. `floyd-warshall` is the clearest case: one static
barrier, executed once per `k`.

### What the two regimes say

The `pragma_form` column splits the suite in two, and read that way the totals
say something the suite-order sum hides:

| block | what it shows |
|---|---|
| `split` (19 kernels) | clang 44, our baseline 44 — *identical, kernel by kernel* — and 25 after the pass |
| `combined` (11 kernels) | clang 0, ours 15, and 0 after the pass |

So the baseline is not a weak one to beat: it is exactly clang's, everywhere
clang does not apply its own elision. What the pass adds is one barrier per
parallel region on the spelling clang's front-end skips.

### Measuring the effect

`BARRIER_ELIM=1` puts the pass in the real pipeline, so the other two drivers
measure its effect. Run each configuration twice, changing only this variable:

```sh
BARRIER_ELIM=0 ./run_performance.sh    # baseline
BARRIER_ELIM=1 ./run_performance.sh    # with the optimisation
BARRIER_ELIM=1 ./run_correctness.sh    # still bit-identical?
```

The two configurations write into separate trees — `results/<runtime>/` for the
baseline and `results/<runtime>-barrier-elim/` for the optimised run — so you
can run them in either order and keep both CSVs. Compare their `opt_par_cyc`
column:

```sh
cd results
awk -F';' '
  BEGIN { printf "  %-24s %14s %14s %9s %s\n", "kernel","base","elim","gain","ref drift" }
  FNR==1 { next }
  NR==FNR { b[$1]=$5; r[$1]=$3; next }
  ($1 in b) && b[$1]+0>0 && $5+0>0 {
    printf "  %-24s %14.0f %14.0f %+8.2f%%  %+9.2f%%\n",
           $1, b[$1], $5, 100*(b[$1]-$5)/b[$1], 100*($3-r[$1])/r[$1]
  }' iomp/results_performance.csv \
     iomp-barrier-elim/results_performance_barrier-elim.csv
```

`gain` above zero is time the pass saved. `ref drift` is the sanity check:
`BARRIER_ELIM` only ever touches the opt cells, so the native column should
come out the same in both runs — where it does not, that is the machine moving
under the measurement, and a gain smaller than the drift means nothing.

Give the timing enough runs to see the difference: removing a handful of
barriers is usually worth a few percent, which `REPS=1` cannot resolve. Keep
`REPS` at its default of 10 (or at least 5) for a comparison you can trust, and
check the `[noisy]` warnings — a cell flagged there is measuring the machine,
not the pass.

### A/B in one run (`BARRIER_ELIM=both`)

On a host runtime the two-run flow above has a problem that no number of `REPS`
fixes: the two configurations are measured **hours apart**, and `REPS` only
averages the noise *inside* a run. Between two runs on this machine the native
cells — the same binary, untouched by the pass — moved by 6% in total and by
far more on individual kernels, which is larger than the effect being looked
for.

`BARRIER_ELIM=both` measures the two configurations against each other inside
one run instead:

```sh
BARRIER_ELIM=both ./run_performance.sh
```

Per kernel it builds `opt_par` twice, with and without the pass, then times
them **alternately** — one repetition of each in turn, so the two are always
seconds apart and anything the machine does lands on both. There is no native
comparison and no sequential cell: they answer a different question, and the
sequential build has no barriers at all (it is compiled without `-fopenmp`), so
it is identical in the two configurations by construction. Two cells per kernel
instead of four means the whole thing costs **half a normal run**, against the
two full runs the split flow needs.

**On `pmsis` it works too**, and it is the way to get the saving reported in
§4.5, which is a claim about the PULP target:

```sh
RUNTIME=pmsis BARRIER_ELIM=both PLOT=true ./run_performance.sh
```

Two things do not carry over there, both of them simplifications. There is no
alternation, because gvsoc has no drift to cancel, and no repetitions, because
a second run of the same binary returns the same cycle count. So the deviation
columns are genuinely `0` rather than unknown — the signs are exact, but each
is a single measurement rather than an average. The driver says so in its
summary and the chart says so on its face, since a zero error bar must not be
allowed to read as a confirmed one.

Output goes to `results/<runtime>-barrier-ab/results_performance_barrier-ab.csv`:

```
kernel;base_par_cyc;base_sd;elim_par_cyc;elim_sd;delta_pct;delta_sd_pct
```

`delta_pct` is the saving, positive when the pass won, and `delta_sd_pct` is
its error bar — the two standard deviations propagated. **A saving smaller than
its own error bar is not a result**, and the summary counts how many kernels
clear twice their error rather than reporting a single headline number.
`PLOT=true` draws it ([`lib/plot_delta.py`](lib/plot_delta.py)): one bar per
kernel with its error bar, and the ones that do not clear it drawn in grey.

Host runtimes only. On `pmsis` gvsoc is deterministic — `REPS` does not even
apply — so two ordinary runs already compare exactly, and the driver says so
rather than pretending the mode is needed.

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
API (fork/barrier/core-id); the harness provides and links those shims.

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
| `POLYBENCH`      | *none — required* | PolyBench-OpenMP checkout root; the drivers refuse to start without it |
| `POLYBENCH_UTIL` | `$POLYBENCH/utilities` | PolyBench support headers           |
| `RULES`          | the repo's `rules.dsl` | DSL file passed to `mlir-opt-omp`   |
| `OUTDIR`         | `$PWD/results` | binaries, dumps and CSVs land in `$OUTDIR/<runtime>`, plus a `-barrier-elim` suffix when `BARRIER_ELIM=1` |

### Run parameters

| Variable  | Default | Meaning                                             |
|-----------|---------|-----------------------------------------------------|
| `RUNTIME` | `iomp`  | `iomp`, `libgomp` or `pmsis` (PULP/gvsoc)           |
| `KERNELS` | *empty* | explicit space-separated kernel list; empty runs the whole suite, paths resolved against `$POLYBENCH` |
| `DATASET` | `MINI_DATASET` for correctness, `LARGE_DATASET` for performance, always `MINI_DATASET` on `pmsis` unless set | PolyBench dataset size macro |
| `THREADS` | `16`    | thread count for the parallel runs; ignored on `pmsis` |
| `BARRIER_ELIM` | `0` | `1` adds `--omp-barrier-elim` to the pipeline and writes to `results/<runtime>-barrier-elim/`. Off by default so a plain run is the baseline to compare against. `both` (perf driver, host only) builds the kernel each way and times the two against each other — see [Measuring the effect](#measuring-the-effect) |

### Performance only (`run_performance.sh`)

| Variable            | Default | Meaning                                   |
|---------------------|---------|-------------------------------------------|
| `REPS`              | `10`     | timed runs per cell, min+max dropped; ignored on `pmsis` (gvsoc is deterministic) |
| `VARIANCE_ACCEPTED` | `5`     | warn if a cell's relative std-dev exceeds this % |

### Plots

One switch for every driver that has a chart to draw: `run_performance.sh` and
the two barrier drivers. Each renders next to its own CSV, so turning it on
once covers all of them.

| Variable            | Default | Meaning                                   |
|---------------------|---------|-------------------------------------------|
| `PLOT`              | `false` | `true` → also render the driver's chart (needs matplotlib) |
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
| `OMP_PLACES`      | *unset* | exported for the parallel runs              |
| `OMP_PROC_BIND`   | *unset* | exported for the parallel runs              |
| `OMP_WAIT_POLICY` | *unset* | keep idle workers spinning — see below      |
| `KMP_BLOCKTIME`   | *unset* | the same for iomp/libomp                    |
| `GOMP_SPINCOUNT`  | *unset* | the same for libgomp                        |

Those five have **no default** — `common.sh` no longer sets them, so a bare run
takes whatever the runtime does on its own. The configs in
[`configs/`](configs/) set them, which is what makes a run under one of them
repeatable; anything else is on you to pin.

The three wait-policy variables are there to make the *timing* repeatable, not
to make it fast. Left at their defaults the runtime parks its workers once a
wait exceeds the block time, and every later region entry then pays a wake-up
syscall per thread — a cost that depends on how the timing happened to fall
rather than on the code under test. Two runs of the **same binary** here
disagreed by 88% on `durbin` and by a factor of ten on `trisolv`, while their
sequential cells repeated to four decimal places and `3mm` — one big region —
repeated to 0.05%. The kernels this ruins are exactly the fine-grained ones,
which are also the ones a synchronisation optimisation shows up on.

Spinning holds a core doing nothing while the region is closed, so it assumes
the machine is yours for the duration. That is the assumption a benchmark run
makes anyway, but it does mean numbers taken with these settings are not
comparable with numbers taken without them.

The strict FP flags are enabled by default and must match between ref and opt —
without them FMA contraction and reordered reductions make iterative kernels
diverge even though both binaries are IEEE-correct. Override only if you know
why.
