# Pre-generated modules

One per `.c` test in the directory above, holding the same program after the
front-end: `clang -fclangir -emit-cir` followed by `cir-opt --cir-to-llvm`. From
here on the pipeline is the same, so running a test from one of these files
exercises everything this repository owns — the DSL evaluation, the outlining,
the plan emission — without needing ClangIR at all.

## Why they exist

ClangIR's OpenMP clause support lives in the EEESlab fork and arrives one clause
at a time. A machine whose LLVM predates a clause cannot compile the `.c` that
uses it, and reports

```
error: ClangIR code gen Not Yet Implemented: OpenMPClause : num_threads
```

which says nothing about the lowering — the thing the test was written to check.
That is not hypothetical: it is what an install from before 2026-05-29 does,
`num_threads` support having landed in the fork on that date.

So the front-end is separable, and `FRONTEND=0` separates it:

```sh
FRONTEND=0 ./run_constructs.sh
```

## What a run from here does and does not prove

It proves the lowering. It does **not** prove the front-end, and the driver says
so on every line rather than letting a green run imply more than it tested:

```
  num_threads   42   42   PASS (pre-generated IR)
```

The default, `FRONTEND=auto`, compiles the `.c` when it can and falls back only
when the front-end refuses — so on an up-to-date machine these files are never
touched, and on an old one the suite still says something true.

## Regenerating

They are generated output, not hand-written, so they are replaced rather than
edited. On a machine whose ClangIR handles every clause:

```sh
for f in ../*.c; do
  n="$(basename "${f%.c}")"
  clang -S -Xclang -fclangir -Xclang -emit-cir -fopenmp \
        -I"$INC_OMP" "$f" -o /tmp/$n.cir
  cir-opt /tmp/$n.cir --cir-to-llvm --reconcile-unrealized-casts -o "$n.mlir"
  sed -i -E 's/cir\.[^,}]+,? ?//g' "$n.mlir"
  sed -i -E "s#^module @\"[^\"]*\"#module @\"$n.c\"#" "$n.mlir"
done
```

The last `sed` matters: the module name otherwise carries the absolute path of
the machine that generated it, which is noise in a diff and a small privacy leak
in a published artifact.

Regenerate when a `.c` changes, or when the front-end starts emitting something
materially different — and check the diff, because that diff is the front-end's
output changing under you, which is worth seeing rather than absorbing.
