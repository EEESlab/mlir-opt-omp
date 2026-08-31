# Lowering complexity analysis (LoC)

The scripts behind the lines-of-code comparison in the paper (Section 4.4,
Table 2) and its supplementary material. They measure the *minimal*
implementation effort a compiler pays to lower the OpenMP constructs the tool
supports — `parallel`, `for` with `schedule(static|dynamic)`, `task` and
`barrier` — in GCC and in Clang.

| Script | Reproduces | Result |
|---|---|---|
| [`gcc_omp_loc.py`](gcc_omp_loc.py) | supplementary Table 1 | 21,761 lines attributed out of 82,400, over 15 files |
| [`clang_omp_loc.py`](clang_omp_loc.py) | supplementary Table 2 | 7,617 lines attributed out of 50,754, over 4 files |

These are measurement scripts, not tests: `lit` does not pick them up and they
are not part of `check-omp`.

## Running them

Each script needs a checkout of the code base it measures, passed with
`--root` or through `$GCC_SRC` / `$LLVM_SRC`. The line numbers they carry are
anchors into a specific commit, so check out that commit first — otherwise the
counts drift and, on the Clang side, the anchors stop naming the entities they
should.

```sh
git -C ~/gcc checkout 113f406e521057894e4cd3af2355f814ad203e9a   # GCC 17.0.0
python3 gcc_omp_loc.py --root ~/gcc

git -C ~/llvm-project checkout 4f92cf9599c4077c08b7fac0a21624e55da572f9  # LLVM 23.0.0git
python3 clang_omp_loc.py --root ~/llvm-project
```

Both print one line per counted entity, then the per-file table the paper
reports. They exit non-zero if an anchor no longer matches, if an entity is not
found, or if two entities overlap — that is, whenever the numbers can no longer
be trusted. A checkout other than the pinned commit, or a per-file count that
differs from the published one, is reported as a warning: the run is still
meaningful, it just no longer reproduces the table.

## Method

Two stages, applied to both code bases.

**Files.** Only the code that implements the lowering itself is in scope: what
turns an OpenMP construct into runtime calls, plus the infrastructure it
depends on. Directive parsing, AST construction and semantic checking are
outside the comparison, because the tool reuses them from the host compiler
rather than reimplementing them. That leaves, in GCC, gimplification and the
`omp-low` / `omp-expand` middle-end passes with their IR and builtin
declarations; in Clang, `SemaOpenMP.cpp` together with `CGStmtOpenMP.cpp` and
`CGOpenMPRuntime.cpp/.h`.

`SemaOpenMP.cpp` is counted because the lines attributed to it implement the
canonical loop-form analysis — the `OpenMPIterationSpaceChecker` class and
`checkOpenMPLoop` — which computes the loop bounds and trip count the code
generator then consumes. GCC does the same work later, in `gimplify.cc` and
`omp-low.cc`, which are counted too. The AST definitions of directives and
clauses stay outside on both sides.

**Entities.** Inside those files, every function, method, class or
data-structure definition is *minimal* if it lies on the compilation path of
the target constructs; only minimal entities contribute. Handlers of other
directives (`sections`, `single`, `target`, …) and of explicit clauses other
than `schedule` are excluded. The list of minimal entities is embedded in each
script, one entry per entity, and is the part to edit when the feature set
under comparison changes.

A few entities are common entry points for every OpenMP directive —
`ActOnOpenMPExecutableDirective` in Clang, the `scan_omp_1_stmt` and
`lower_omp_1` dispatchers in GCC. Only a small part of their body serves the
target constructs, but they are needed to support them, so they are counted
whole.

Each entity is measured from its leading comment down to its closing brace, by
brace-matching over lines with comments and string literals blanked out;
declarations are measured to the terminating `;`, and a few IR and builtin
definitions by a fixed line range. Blank lines and comments inside a region
count, uniformly across the code bases. Every script checks that no line is
attributed twice.

## The mlir-opt-omp side

Table 3 of the supplementary material counts every source file of this
repository, 5,040 lines at the camera-ready snapshot: the entry point (99), the
build configuration (100), the dialect (247), the DSL engine (1,294), the four
passes (2,985) and [`rules.dsl`](../../rules.dsl) (315). No script embeds that
list, since it is a plain `wc -l` over the whole tree:

```sh
wc -l rules.dsl $(git ls-files '*.cpp' '*.h' '*.td' 'CMakeLists.txt')
```

Run today it gives a larger figure than the published one: the tool has since
gained the barrier-elimination pass, and the build configuration is now split
across per-directory `CMakeLists.txt` files.
