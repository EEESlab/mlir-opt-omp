# Reference results

The values the paper's figures plot and the numbers its text states, so a run
can be compared against something instead of being read in isolation.

The two CSVs here divide the work strictly, and the split is the point:

| file | holds | comes from |
|---|---|---|
| [`reference.csv`](reference.csv) | what the figures **plot** | read back out of the vector files, exact |
| [`claims.csv`](claims.csv) | what the text **says** | typed in from the paper, one row per sentence |

Where the two disagree the paper has a problem, and keeping them apart is what
makes that visible instead of averaging it away — the 3% mean under *What it
reproduces* is the case in point.

## What is here

| file | figure | holds |
|---|---|---|
| `results_gomp_LARGE_FINAL.eps` | 4 | parallel speedup, libgomp: GCC frontend vs ours |
| `results_iomp_LARGE_FINAL.eps` | 5 | parallel speedup, libomp: Clang frontend vs ours |
| `results_pulp.eps` | 6 | parallel speedup on GAP8: PULP-SDK GCC vs ours |
| `results_pulp_sizes.eps` | 7 | binary size change on GAP8 |
| `unroll_speedup.eps` | 8 | the CIR unroll-by-two gain |
| [`reference.csv`](reference.csv) | — | all five, as numbers |
| [`claims.csv`](claims.csv) | — | the sentences with a number in them, §4.2–§4.5 |
| [`extract_from_eps.py`](extract_from_eps.py) | — | how the numbers came out of the figures |

## The numbers are exact

`reference.csv` is not a transcription and not an estimate. The figures are
matplotlib EPS, which is vector: every bar is a rectangle with real
coordinates, and every axis tick is drawn at its own position with its own
label beside it. The extractor fits the device-to-data transform from the
ticks and reads each bar back through it, so what lands in the CSV is what the
figure plots, to four decimal places.

The tick's position comes from the tick *mark*, not from the label next to it.
A tick label is centred on its tick, so in the file it sits half a text height
lower -- a constant 0.3415 em, identical at the top of the axis and at the
bottom. Taking the origin from the label leaves the ticks perfectly collinear
and the scale exactly right, and shifts every value by that one constant:
+0.42 on figure 4, +0.66 on figure 8. Nothing about the resulting numbers looks
wrong, which is why the reading is now anchored to the marks and checked
against the baseline.

Regenerate it, or check it still matches, with:

```sh
python3 extract_from_eps.py            # rewrite reference.csv
python3 extract_from_eps.py --check    # exit non-zero if it drifted
```

Five things are asserted while reading rather than assumed, because each would
silently corrupt the result: that the y ticks are collinear (a log axis read as
linear gives plausible nonsense), that there are as many tick marks as tick
labels (otherwise the values are paired to the wrong positions), that **the
bars read as standing on zero** (a bar chart's baseline is zero, so this is the
one thing the transform can be checked against that the figure did not itself
supply -- it is what catches an origin taken from the wrong feature), that
every figure has exactly as many labels as bar groups (otherwise the pairing is
guesswork), and that a bar below the baseline becomes a negative value rather
than a positive one.

## What it reproduces

The extracted values agree with what the paper states in prose, which is the
check that the extraction is right — with one exception, noted below:

| the paper says | the CSV says |
|---|---|
| §4.2 `doitgen` is faster with our flow on libgomp | 8.49 native, **10.11** ours |
| §4.3 `floyd-warshall`, `deriche`, `nussinov` are behind on PULP | 7.22/6.33, 7.16/6.04, 6.52/5.61 — all three |
| §4.3 the size increase stays below 0.7% | highest is `nussinov` at **0.6826%** |
| Fig. 8 about 24% on `floyd-warshall` | **23.57%** |

Two further checks come free, and neither was put there on purpose. `seidel-2d`
is the one kernel PolyBench leaves serial, and it reads **1.0000** on figure 4
and 0.9986/1.0003 on figure 5 — a speedup of exactly one, recovered rather than
assumed. And the kernel order recovered from the figures is identical to
`ALL_KERNELS` in `../lib/kernels.sh`, which is an independent check that the
bars were paired to the right names.

One claim does not survive, and it is not the one that used to fail here. Fig. 8
is described as about 3% on average over ten applications; the ten bars average
**2.28%**. The 0.7% size bound in §4.3 *does* hold — an earlier version of this
file reported it broken, at 0.7296% on `nussinov`, which was the constant offset
described above and not the paper.

## Which claims have a checker

`claims.csv` carries a `checked_by` column naming the driver that reads each
row, and `-` where nothing does. That is deliberate: the file is an inventory
of what the paper asserts, not only of what happens to be testable today, and
`compare_to_reference.py` prints the uncovered rows for the runtime it was
given as its last section.

| claim | checked by |
|---|---|
| §4.2/§4.3 named kernels, §4.3 size bound | `lib/compare_to_reference.py` |
| §4.5 barrier counts (59/26/45/28) | `run_barrier_vs_native.sh`, full suite only |
| §4.5 barrier saving, kernels improving, range | `run_performance.sh`, `BARRIER_ELIM=both` on pmsis |
| Fig. 8 unroll gains (3%, 24%), and each bar of the figure | `run_unroll.sh` |

Every row has a driver now. `run_unroll.sh` is the one that cannot run
everywhere: the unroll-by-two pass is a CIR pass, so it is in the ClangIR
fork and not in this repository, and the driver refuses rather than
reporting a delta of zero when the `cir-opt` it is given does not have it.

The Table 2 line counts are checked where they are produced, against the
per-file constants in [`../../LoC/`](../../LoC/).

## Reading a run against these

`run_performance.sh` prints the comparison itself when it finishes. By hand:

```sh
python3 ../lib/compare_to_reference.py \
  ../results/libgomp/results_performance.csv --runtime libgomp
```

It prints one line per kernel: the run's own two speedups and their ratio, the
two the figure plots and the ratio against those, and on `pmsis` the size
change here and in Figure 7. Then every claim about that runtime with the run's
number beside it. It labels nothing as close, acceptable or at parity — the
columns are put side by side and the reading is the reviewer's.

Worth knowing while reading it: these figures were measured on the machine in
§4.1, and an absolute speedup does not survive a change of CPU — a kernel
reaching 8x there and 5x on a reviewer's machine means nothing is wrong. That
is why the native bar is printed too: if it moved by the same factor, the
machine moved, not the compiler.

## Configuration the numbers assume

The host figures were taken at `THREADS=16` and `DATASET=LARGE_DATASET`,
matching §4.1. Thread binding and wait policy were left at the runtime's own
defaults — the configs under [`../configs/`](../configs/) carry the rest, and
the commented-out lines there record an attempt to pin them that did not deliver
the repeatability it promised.

The PULP figures have no such caveat: gvsoc is deterministic, and none of those
variables applies to a bare-metal runtime.

`compare_to_reference.py` enforces the dataset half of this rather than
trusting it: `LARGE_DATASET` on the host, `MINI_DATASET` on `pmsis`, and the
barrier pass off, since Figures 4–7 were measured without it. A run outside
that says so and compares nothing — a different dataset is a different problem
size, and putting its speedups next to these columns would be comparing two
different programs. The thread count is reported beside the figure's rather
than enforced.
