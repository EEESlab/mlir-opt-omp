# Reference results

What a run of `run_performance.sh` is expected to look like, so a result can be
compared against something instead of being read in isolation.

## Status: provisional

[`expected-from-paper.csv`](expected-from-paper.csv) holds speedups **digitised
by eye from Figures 4, 5 and 6** of the paper. It is not a measurement, and it
is not precise: about ±0.5 on the Figure 4 and 5 axes, ±0.3 on Figure 6.

Use it for the question it can answer — *does this run resemble the paper at
all?* Is `cholesky` the libgomp outlier, does `doitgen` come out ahead for us,
are `lu` / `ludcmp` / `trisolv` / `durbin` flat near 1. Do not use it to judge a
difference of a few percent: at that scale the file's own error is larger than
what you are measuring.

## Why it should be replaced

Two reasons, and the second is the serious one.

Reading pixels off a chart loses precision that the original CSVs still have —
`run_performance.sh` always writes them, they were simply never kept.

More importantly, **the published figures predate the compiler in this repo**.
Commits `bb13331` and `015587d` (2026-08-27) moved every alloca into its
function's entry block. That changes the code generated for every wsloop, and
it was motivated by a PULP bug — `seidel-2d` overrunning the cluster stack on
GAP8. Figure 7 is the clearest case: it measures linked-ELF size, and moving
the allocas changes the frame setup, so the bytes move with it.

So the numbers here describe a compiler slightly different from the one a
reviewer will build.

## Replacing it

Re-run the three configurations on the reference machine with the current
branch built, and keep the CSVs this time:

```sh
cd test/Integration
RUNTIME=libgomp SUITE=full DATASET=LARGE_DATASET PLOT=true ./run_performance.sh   # Fig 4
RUNTIME=iomp    SUITE=full DATASET=LARGE_DATASET PLOT=true ./run_performance.sh   # Fig 5
RUNTIME=pmsis   SUITE=full                       PLOT=true ./run_performance.sh   # Fig 6 + 7
```

Copy each `results/<runtime>/results_performance.csv` in here, and regenerate
the paper's figures from those same files — then chart, CSV and paper agree by
construction rather than by trust. Record alongside them: machine, commit hash,
date, `THREADS`, `DATASET`, and the wait-policy variables below. A CSV without
the configuration it was taken under cannot be compared against.

## Configuration the numbers assume



Run them with:

```sh
cd ..
RUN_ENV=configs/paper-libgomp.env PLOT=true ./run_performance.sh   # Fig 4
RUN_ENV=configs/paper-iomp.env    PLOT=true ./run_performance.sh   # Fig 5
RUN_ENV=configs/paper-pmsis.env   PLOT=true ./run_performance.sh   # Fig 6 + 7
```

The host configs pin `THREADS=16` and `DATASET=LARGE_DATASET`, matching §4.1 of
the paper ("16 hardware threads", "the large configuration"). For a shorter run
add `REPS=5` (halves it, keeps a weak deviation) or `REPS=3` (3.3× faster, the
median of three, no deviation) on the command line — the dataset is
deliberately not the knob, since a smaller one changes which value you are
measuring rather than how confident you are in it.

## Numbers the paper states exactly

These are prose, not chart pixels, so they are quoted rather than digitised —
and they are the ones to check a run against when precision matters.

| Claim | Value | Where it comes from |
|---|---|---|
| Binary size increase on PULP, ours vs sequential | **below 0.7%** in every case | `run_performance.sh` `RUNTIME=pmsis`, the `size_*` columns (Figure 7) |
| Team-barrier call sites, ours without the pass | **59** | `run_barrier_vs_native.sh` |
| …with `--omp-barrier-elim` | **26** | ” |
| …clang, same kernels | **45** | ” |
| …gcc, counted at `-O0` | **28** | ” |
| Run-time saving from the pass on GAP8 | **0.037%** overall; 22 of 30 kernels improve; per-kernel −0.16% … +1.12% (`trisolv`) | `RUNTIME=pmsis BARRIER_ELIM=both ./run_performance.sh` |
| CIR unroll-by-two gain | **~3%** average over ten kernels, **~24%** on `floyd-warshall` | Figure 8; the pass lives in the ClangIR fork, not in this repo |
