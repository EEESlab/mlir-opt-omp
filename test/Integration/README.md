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
- **`run_unroll.sh`** — what the CIR unroll-by-two pass is worth on GAP8,
  which is Figure 8 (see [CIR unrolling](#cir-unrolling)). `pmsis` only,
  and only with a `cir-opt` that carries the pass.

A third driver covers the constructs and clauses PolyBench never writes — which
is most of them (see [`constructs/`](constructs/)):

- **`constructs/run_constructs.sh`** — one standalone C program per
  construct/clause, each printing `42` only if the clause actually did its job.
  Across the whole 30-kernel suite PolyBench uses `parallel`, a bare `for` and
  `private`, and nothing else: no `firstprivate`, `num_threads`, `proc_bind`,
  `nowait`, explicit `schedule`, `if`, `barrier`, `task` or `taskwait`. Without
  these files those cells have no end-to-end coverage at all.

  The oracle is the printed value, not a diff against the reference compiler: a
  clause both compilers ignored would agree and pass. Each program observes the
  effect itself — the team really has the requested size, the copy really was
  taken at entry, every iteration really ran once. Two of them (`nowait`,
  `proc_bind`) cannot observe their own clause portably and say so in their
  headers; their deterministic evidence is in `../Regression/`.

  Run: `constructs/run_constructs.sh` (or one test by name, `... num_threads`).

  **The IR of every stage is kept**, under
  `results/<runtime>/constructs/<test>/`, because it is the point of these
  tests as much as the verdict is: a passing line says the clause works,
  `02-lowered.mlir` says what it turned into. That file is the one to read —
  the clause has become runtime calls and no generic MLIR pass has run over
  them yet, so `num_threads` shows up as `__kmpc_push_num_threads` and
  `if(0)` as the `__kmpc_serialized_parallel` pair. `KEEP=0` discards them.

  **If ClangIR is older than the clauses**, the `.c` files stop compiling long
  before the lowering is reached — `Not Yet Implemented: OpenMPClause :
  num_threads` and so on — and the failure says nothing about the part this
  repository owns. A copy of each module after the front-end is checked in
  under `constructs/mlir/`, so that part can be tested anyway:

  ```sh
  FRONTEND=0 constructs/run_constructs.sh
  ```

  The default (`FRONTEND=auto`) compiles the `.c` when it can and falls back
  only when the front-end refuses. A run that fell back says so on every line —
  it proved the lowering, not the front-end, and those are different claims.

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
    <kernel>-omp/<kernel>-omp_ref.dump
    <kernel>-omp/<kernel>-omp_opt.dump
```

The two dumps are the evidence and the only thing kept: the result *is* that
they are identical, so a reader can run the diff themselves instead of taking
the verdict on trust. The binaries and the IR beside them are build products,
rebuilt by re-running, and `KEEP=1` holds on to them when a FAIL has to be
chased. `run_performance.sh` prunes the same way and keeps only its CSV — the
cycle counts are the result, and the mean and deviation the per-repetition logs
produced are in that CSV already.

`constructs/` deliberately does the opposite and keeps every stage: there the
question is *how* a clause is lowered and the IR is the answer, where here the
question is whether two programs agree and the dumps are.

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

It only runs when this run was made the way the figure was, and it is handed
the configuration to decide that:

| runtime | compares only at | because |
|---|---|---|
| `libgomp`, `iomp` | `LARGE_DATASET` | §4.1; Figures 4 and 5 |
| `pmsis` | `MINI_DATASET` | §4.1, and `common.sh` forces it anyway |
| any | `BARRIER_ELIM=0` | Figures 4–7 were measured without the pass |

Anything else prints one line saying which of those it was and stops — a
different dataset is a different problem size, so the published columns would
not be about the same program. `THREADS` is **not** a precondition: it changes
the speedups but not what they are speedups of, so a run at 8 threads is
compared and the header says `threads 8 (figure: 16)`.

It **reports and does not judge**. No row is labelled close, acceptable, at
parity or reversed: the columns are put next to each other and what they mean
is the reader's call. One line per kernel:

```
kernel            run_nat  run_opt  run_o/n fig6_nat fig6_our fig6_o/n    size%    fig7%
nussinov             6.63     6.65    1.003     6.52     5.61    0.860    0.045    0.683
```

The prefix says where a number came from — `run_` measured here, `fig<n>_` read
off the paper — and the suffix says which variant it is, the native compiler or
this tool's (`opt`, the name the CSV columns already use: `speedup_opt`,
`opt_vs_native`). Each source gets the same three columns, and the third is the
one to read: the two `_o/n` ratios are the same quantity measured twice, once
here and once by the paper. They are the comparison, and they are the two
columns printed in colour on a terminal.

| column | is |
|---|---|
| `run_nat`, `run_opt` | measured by this run: the speedup of each variant against **its own** sequential cell |
| `run_o/n` | `run_opt / run_nat` — this tool against the native compiler, here |
| `fig<n>_nat`, `fig<n>_our` | what the paper's figure plots for that kernel, from [`reference/reference.csv`](reference/reference.csv) |
| `fig<n>_o/n` | `fig<n>_our / fig<n>_nat` — the same ratio, in the figure |
| `size%`, `fig7%` | `size_opt_par / size_opt_seq - 1`, here and in Figure 7 — `pmsis` only |

Why the ratios and not the speedups: an absolute speedup does not survive a
change of CPU, so `run_opt` sitting well above `fig<n>_our` says the reviewer's
machine is faster and nothing about the tool — the row above is a `pmsis` run
where every kernel scales better than the paper's. A ratio divides that out on
both sides, which is what makes the two `_o/n` columns comparable at all. The
example is nussinov, one of the three kernels §4.3 names as behind: 0.860 in
Figure 6, 1.003 here.

Both `_o/n` columns get a geomean, and the absolute `opt_vs_native` geomeans
follow on one line — parallel and sequential, which is the pair that says
whether a deficit is uniform.

The table is the whole output: no claim list, no verdict column. Section 4.5 is
checked by the drivers that measure it — `run_barrier_vs_native.sh` and
`run_unroll.sh` — each against [`reference/claims.csv`](reference/claims.csv),
where the number is one those runs actually produce.

Nothing here exits non-zero.

One more thing worth knowing while reading the table, which the script will not
decide for you: a backend that emits slower code shows up in `opt_vs_native`
while `run_o/n` stays near 1.000, because each speedup is taken against that
compiler's own sequential run, so a uniform deficit sits in both halves of the
fraction and cancels. The ratio says the parallelisation is as good; it does not
say the code is as fast.

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
./run_barrier_vs_native.sh   # -> results/barrier_vs_native/results_barrier_vs_native.csv
```

There is no runtime to pick: this is a count of barriers in two runtimes at
once, ours and clang's against the iomp ABI (`__kmpc_barrier`), gcc's against
its own (`GOMP_barrier`). `RUNTIME` is therefore ignored wherever it is set —
`run.env`, `RUN_ENV` or the command line — while everything else those files
carry (`POLYBENCH`, the tool paths, `DATASET`, `KERNELS`, `OUTDIR`) still
applies, and the results go to one directory of their own rather than under a
runtime.

The `form` column says how the kernel spells its parallel loop: `combined` for
`#pragma omp parallel for`, `split` for a `#pragma omp parallel` region with a
separate `#pragma omp for` inside. Clang elides the trailing barrier only for
the combined directive, which is where the delta comes from.

On a full-suite run the four totals are printed next to what §4.5 states — 59
without the pass, 26 with it, 45 for Clang, 28 for GCC at `-O0` — read from
[`reference/claims.csv`](reference/claims.csv). Side by side and nothing more.
A subset run, or one where a kernel failed to build, says why it is not
printing them rather than showing a short sum.

Every file a number was counted in is kept, one directory per kernel, so the
table can be rechecked instead of taken on trust:

```
results/barrier_vs_native/gemm-omp/
  clang-O3.ll  ours-baseline-O3.ll  ours-elim-O3.ll  gcc-O0.s
```

Each count is one grep over one of those files, and the run prints both forms
when it finishes:

```sh
grep -o 'call void @__kmpc_barrier' ours-elim-O3.ll | wc -l
grep -cE '\b(call|jmp)\b.*GOMP_barrier' gcc-O0.s
```

The directory is wiped at the start of each run, so what is in it always
belongs to the CSV beside it.

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
its own, counted at `-O0`**:

```sh
gcc -fopenmp -O0 -S      # counting GOMP_barrier call sites
```

gcc decides the elision in the front end, so it is already applied at `-O0`,
and nothing has yet copied a call site. From `-O1` on the count rises without
any execution gaining a barrier: paths get duplicated and each copy carries the
barrier sequence — over this suite 28 becomes 43 at `-O3`, gemver alone 3
becomes 7. That number would measure the shape of the CFG as much as
synchronisation.

Jump threading is most of it but not all of it, and how much depends on the gcc
build: on one toolchain `-fno-thread-jumps` put every kernel back on its `-O0`
number, on another the suite still came out at 41. So the stage is what makes
the column readable, not a flag. `-O0` it is — and say which stage when quoting
the column, since the other three are after `-O3`. Our own count does not move
between stages: the same 59/25 in MLIR and after `-O3`.

The duplication is *not* loop cloning: `GCC_STRICT_FP` already turns the
vectoriser off, so no kernel gets a vector and a scalar copy. Four `omp for` in
one region are enough to see the effect on its own — 3 call sites at `-O0`, 6
from `-O1`, 3 again with `-fno-thread-jumps`.

Count `GOMP_barrier`, not `call GOMP_barrier`: from `-O2` gcc emits the last
barrier of a region as a tail call, which prints as `jmp` and which an anchor
on `call` silently drops. It does not arise at `-O0`, but the driver's pattern
matches both mnemonics anyway, so a count taken at another stage stays honest.

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

On a full-suite `pmsis` run the summary is followed by the four aggregates §4.5
states — the 0.037% saving, 22 of 30 kernels improving, and the two ends of the
per-kernel range — read against
[`reference/claims.csv`](reference/claims.csv). The kernel currently holding
the best and the worst delta is printed beside them, because the paper names
`trisolv` at the top of that range and a different name there means the
sentence needs one too.

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

## CIR unrolling

Figure 8 is the other half of §4.5: what a CIR-level unroll-by-two is worth on
GAP8, and the case for keeping OpenMP in MLIR long enough that a CIR pass can
still run underneath it.

```sh
RUNTIME=pmsis ./run_unroll.sh     # -> results/pmsis/results_unroll.csv
```

Per kernel it builds the parallel binary twice — with the pass and without —
runs both on gvsoc and reports the cycle saving. One run per cell: the
simulator is deterministic.

**The pass is not in this repository.** It is a CIR pass, so it belongs to the
ClangIR fork whose `cir-opt` this harness calls, and at the time of writing no
`cir-opt` here carries it. The driver looks for it in `cir-opt --help`, and
when it is missing it refuses and lists what that build does offer, rather than
running two identical binaries and reporting a saving of zero. If the pass is
spelled in a way the search misses, name it:

```sh
CIR_UNROLL_PASS=--your-pass ./run_unroll.sh
```

By default it runs the **11 kernels Figure 8 plots**, and it gets that list
from `fig8_unroll_pct` in [`reference/reference.csv`](reference/reference.csv)
rather than from a second copy — the figure decides its own kernel set. `ALL=1`
runs the whole suite; the paper says the remaining kernels are only marginally
affected, which is a claim `ALL=1` is the way to test.

Two comparisons come out at the end: the two sentences of §4.5 (about 3% across
the ten applications, about 24% on `floyd-warshall`) from
[`reference/claims.csv`](reference/claims.csv), and then every kernel against
the bar Figure 8 draws for it. The second is worth doing per kernel rather than
in aggregate — unlike a host speedup, these were measured on a simulator anyone
can rerun, so a divergence is a real difference and not the hardware.

Both are printed as numbers only: measured, published, and the difference. No
verdict column. The tolerance in `claims.csv` says how precisely the paper
states a quantity, which is not a threshold this driver is entitled to rule
against, and a row marked passed or failed would put a judgement in front of
the reviewer before the numbers it came from.

What it measures is the change in **parallel run time**, not in the
parallel-speedup ratio. §4.5 calls it a "speedup increment", which admits both
readings; unrolling also speeds up the sequential build, so in a ratio of two
speedups each taken against its own sequential cell the gain would largely
cancel and 24% could not survive. The driver's header states this reading
explicitly, and if it is wrong the per-kernel comparison against Figure 8 is
where it will show.

Both `.ll` files are kept per kernel, so what the pass did is a diff away:

```
results/pmsis/unroll/<kernel>-omp/
  <kernel>-omp_omp-on.ll           without the pass
  <kernel>-omp_omp-on_unrolled.ll  with it
```

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
| `PULP_KEEP_TMP`  | `0`     | `1` → keep the intermediate `.cir`/`.mlir` of each cell |
| `CIR_UNROLL_PASS` | *auto-detected* | `run_unroll.sh` only: the `cir-opt` unrolling pass. Found in `cir-opt --help` when it is there; set this when it is spelled unexpectedly |
| `ALL`            | `false` | `run_unroll.sh` only: run the whole suite instead of the 11 kernels Figure 8 plots |

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
