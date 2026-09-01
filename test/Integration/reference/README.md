# Reference results

The values the paper's figures plot and the numbers its text states, so a run
can be compared against something instead of being read in isolation.

The two CSVs here divide the work strictly, and the split is the point:

| file | holds | comes from |
|---|---|---|
| [`reference.csv`](reference.csv) | what the figures **plot** | 
| [`claims.csv`](claims.csv) | what the text **says** | 


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



## Reading a run against these

`run_performance.sh` prints the comparison itself when it finishes. By hand:

```sh
python3 ../lib/compare_to_reference.py \
  ../results/libgomp/results_performance.csv --runtime libgomp
```

It prints one line per kernel: the run's own two speedups and their ratio, the
two the figure plots and the ratio against those, and on `pmsis` the size
change here and in Figure 7.

## Configuration the numbers assume

The figures were taken at `THREADS=16` and `DATASET=LARGE_DATASET` for iomp e libgomp,
matching §4.1. 

For PULP `DATASET=MINI_DATASET`

