# Reference results

The values the paper's figures plot and the numbers its text states, so a run
can be compared against something instead of being read in isolation.

The two CSVs here divide the work strictly, and the split is the point:

| file | holds | comes from |
|---|---|---|
| [`reference.csv`](reference.csv) | what the figures **plot** | read back out of the vector files, exact |
| [`claims.csv`](claims.csv) | what the text **says** | typed in from the paper, one row per sentence |

Where the two disagree the paper has a problem, and keeping them apart is what
makes that visible instead of averaging it away — the 0.7% bound under *What it
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
coordinates, and every axis tick carries both its position and its label. The
extractor fits the device-to-data transform from the ticks and reads each bar
back through it, so what lands in the CSV is what the figure plots, to four
decimal places.

Regenerate it, or check it still matches, with:

```sh
python3 extract_from_eps.py            # rewrite reference.csv
python3 extract_from_eps.py --check    # exit non-zero if it drifted
```

Three things are asserted while reading rather than assumed, because each would
silently corrupt the result: that the y ticks are collinear (a log axis read as
linear gives plausible nonsense), that every figure has exactly as many labels
as bar groups (otherwise the pairing is guesswork), and that a bar below the
baseline becomes a negative value rather than a positive one.

## What it reproduces

The extracted values agree with everything the paper states in prose, which is
the check that the extraction is right:

| the paper says | the CSV says |
|---|---|
| §4.2 `doitgen` is faster with our flow on libgomp | 8.91 native, **10.52** ours |
| §4.3 `floyd-warshall`, `deriche`, `nussinov` are behind on PULP | 7.45/6.56, 7.39/6.27, 6.76/5.84 — all three |
| Fig. 8 about 3% on average over ten applications | **2.95%**, excluding `floyd-warshall` |
| Fig. 8 about 24% on `floyd-warshall` | **24.24%** |

The kernel order recovered from the figures is identical to `ALL_KERNELS` in
`../lib/kernels.sh`, which is a second, independent check that the bars were
paired to the right names.

One claim does not survive. §4.3 says the binary size increase "remains below
0.7% in all instances"; the figure it refers to has `nussinov` at **0.7296%**
and `jacobi-2d` at **0.7096%**. The bound is real, but it is 0.75%, not 0.7%.

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

Note what that comparison can and cannot tell you. These figures were measured
on the machine in §4.1, and an absolute speedup does not survive a change of
CPU — a kernel reaching 8x there and 5x on a reviewer's machine means nothing is
wrong. What does transfer is the relationship between the two bars, which is a
property of the compiler and is what the paper actually claims. The comparison
is ordered accordingly: the checks that survive a change of machine come first,
and the absolute one is reported last and labelled as orientation.

## Configuration the numbers assume

The host figures were taken at `THREADS=16` and `DATASET=LARGE_DATASET`,
matching §4.1. Thread binding and wait policy were left at the runtime's own
defaults — the configs under [`../configs/`](../configs/) carry the rest, and
the commented-out lines there record an attempt to pin them that did not deliver
the repeatability it promised.

The PULP figures have no such caveat: gvsoc is deterministic, and none of those
variables applies to a bare-metal runtime.
