# Lowering complexity analysis (LoC)

The two scripts behind the lines-of-code comparison of Section 4.4: how much of
GCC and of Clang a *minimal* lowering of `parallel`, `for` with
`schedule(static|dynamic)`, `task` and `barrier` has to carry.

| Script | Reproduces | Result |
|---|---|---|
| [`gcc_omp_loc.py`](gcc_omp_loc.py) | supplementary Table 1 | 21,761 of 82,400 lines, 15 files |
| [`clang_omp_loc.py`](clang_omp_loc.py) | supplementary Table 2 | 7,617 of 50,754 lines, 4 files |

Not tests: `lit` ignores them and `check-omp` does not run them.

## Running them

Plain Python 3, stdlib only. They read source text — no build, no toolchain,
same numbers on any machine — but need the two code bases at the pinned
commits:

```sh
git clone --filter=blob:none https://github.com/gcc-mirror/gcc.git ~/gcc
git -C ~/gcc checkout 113f406e521057894e4cd3af2355f814ad203e9a
python3 test/LoC/gcc_omp_loc.py --root ~/gcc

git clone --filter=blob:none https://github.com/llvm/llvm-project.git ~/llvm-project
git -C ~/llvm-project checkout 4f92cf9599c4077c08b7fac0a21624e55da572f9
python3 test/LoC/clang_omp_loc.py --root ~/llvm-project
```

`$GCC_SRC` / `$LLVM_SRC` work in place of `--root`. Each script prints one line
per counted entity, then the per-file table the paper reports, and exits
non-zero when the numbers can no longer be trusted: an anchor that no longer
names its entity, an entity not found, two entities overlapping. A checkout
other than the pinned commit, or a count that differs from the published one,
is only a warning.

Cloning both is a few GB, for 19 files that are under 1 MB compressed.
`--list-files` prints what each script reads, so a snapshot of just those works
as a root, and is small enough to ship in the review image (with `$GCC_SRC`
pointing at it):

```sh
mkdir -p loc-src/gcc
git -C ~/gcc archive 113f406e521057894e4cd3af2355f814ad203e9a \
    $(python3 test/LoC/gcc_omp_loc.py --list-files) | tar -x -C loc-src/gcc
python3 test/LoC/gcc_omp_loc.py --root loc-src/gcc
```

## What is counted

Full rationale in Section 1 of the supplementary material. In short:

- **Scope**: the code that turns a construct into runtime calls, plus the
  infrastructure it depends on. In GCC, gimplification and the `omp-low` /
  `omp-expand` passes with their IR and builtin declarations; in Clang,
  `SemaOpenMP.cpp` and the two code-generation layers. Parsing, AST
  construction and semantic checking are out: the tool reuses them from the
  host compiler.
- `SemaOpenMP.cpp` is in because the lines attributed to it are the canonical
  loop-form analysis, which computes the bounds and trip count the code
  generator consumes. GCC does that later, in `gimplify.cc` and `omp-low.cc`,
  counted too.
- **Entities**: a function, method, class or struct counts if it lies on the
  compilation path of the four constructs. Handlers of other directives and of
  clauses other than `schedule` do not. The full list is in the scripts.
- Dispatchers shared by every directive (`ActOnOpenMPExecutableDirective`,
  `scan_omp_1_stmt`, `lower_omp_1`) are counted whole, though only part of them
  serves these constructs.
- Each entity spans its leading comment to its closing brace, blank lines and
  internal comments included, uniformly on both sides. The scripts check that
  no line is attributed twice.

Table 3, the 5,040 lines of this repository, is a plain `wc -l` over the tree
at the camera-ready commit; today it reads higher, the tool having gained the
barrier-elimination pass since.
