# Tests

Two layers, following the LLVM/MLIR convention.

## Regression tests (`Regression/`) — lit + FileCheck

Tests of the IR transforms. 
Each `.mlir` file embeds the command to run (`// RUN:`) and the expected output (`// CHECK:`). 
No C compiler or OpenMP runtime is involved, they exercise the `mlir-opt-omp` passes directly on
hand-written IR
Are the right place to lock in behaviour for every new feature. 

Mirrors `mlir/examples/standalone/test/` in llvm-project, the canonical
out-of-tree MLIR test setup.

### Layout

One directory per construct, one subdirectory per clause. Tests that exercise a
construct as a whole rather than one clause sit directly in the construct
directory; `dialects.mlir` covers no construct and stays at the root.

```
Regression/
  parallel/   if/  num_threads/  proc_bind/  private/  firstprivate/
  wsloop/     schedule-static/  schedule-dynamic/  nowait/
  barrier/
  task/       if/  firstprivate/
  taskwait/
```

lit recurses, so a test is picked up wherever it lands. **Clause directories are
kept even when empty** — they are the checklist of what is supported but not yet
covered, so an empty one is a gap to fill rather than a directory to delete.
Git does not track empty directories, hence the `.gitkeep` in each.

### Coverage

Clauses supported by the lowering, and whether a regression test covers them:

| Construct | Clause | iomp | libgomp | pmsis |
|---|---|:--:|:--:|:--:|
| `parallel` | — | ✓ | ✓ | ✓ |
| | `if` | ✓ | ✓ | ! |
| | `num_threads` | ✓ | ✓ | ✗ |
| | `proc_bind` | ✗ | ✗ | ✗ |
| | `private` | — | — | — |
| | `firstprivate` | — | — | — |
| `wsloop` | — | ✓ | ✓ | ✓ |
| | `schedule(static)` | ✓ | — | — |
| | `schedule(dynamic)` | ! | ✗ | ✗ |
| | `nowait` | ✓ | ✓ | ✓ |
| `barrier` | — | ✓ | ✓ | ✓ |
| `task` | — | ✓ | ✓ | n/a |
| | `if` | ✓ | ✓ | n/a |
| | `firstprivate` | ✓ | ✓ | n/a |
| `taskwait` | — | ✓ | ✓ | n/a |

`✓` a test exists, `—` supported but untested, `n/a` the runtime has no such
construct (`rules.dsl` declares no `task`/`taskwait` for pmsis), `!` the clause
is *not* supported and a test asserts the diagnostic rather than a lowering,
`✗` not supported **and not diagnosed** — the clause is accepted and silently
dropped or mislowered. Every `✗` is a latent wrong-code path:

- **`proc_bind`** is declared only in the iomp DSL, and even there it is broken:
  the clause reaches the plan as the *string* `"close"`/`"spread"`, and the
  outlining pass resolves unknown string tokens to `llvm.mlir.undef : !llvm.ptr`
  — so `__kmpc_push_proc_bind` is handed an undef pointer where it expects an
  i32 enum. libgomp and pmsis never mention the clause at all.
- **`num_threads` on pmsis** is ignored: `ext_pi_cl_team_fork` is called with a
  team size hardcoded to 8 in `rules.dsl`.
- **`schedule(dynamic)` on pmsis** matches the *unguarded* `construct wsloop`
  and is lowered as a static block distribution. iomp and libgomp guard theirs
  with `when schedule == static`, so there it fails loudly instead (`!`).

### Encoding a gap

A green suite must not imply a feature exists, so unimplemented behaviour is
recorded one of two ways depending on what the tool does today:

- **Rejected cleanly** — name the test `*-unsupported-*.mlir` and assert the
  diagnostic with `-verify-diagnostics`. It passes, and what it locks in is a
  real property: the tool refuses the input instead of mislowering it. Grep for
  `-unsupported-` to list these.
- **Silently wrong** — write the test against the behaviour the tool *should*
  have and mark it `// XFAIL: *`. It fails, lit counts it under
  `Expected Failures`, and the gap stays visible in every run. When the fix
  lands the test XPASSes, which lit reports as a failure — that is the prompt to
  drop the `XFAIL` line.

Never delete a failing test for an unimplemented feature: an `XFAIL` is the
record that the gap is known, a missing test is not.

### Running

From the build directory:

```sh
cmake --build . --target check-omp
```

### Adding a test

Create `Regression/<construct>/<clause>/<name>.mlir`:

```mlir
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower | FileCheck %s

func.func @foo() {
  omp.barrier
  return
}

// CHECK: omp_lower.construct
// CHECK-SAME: __kmpc_barrier
```

Substitutions provided by `lit.cfg.py`:

| Token        | Expands to                                            |
|--------------|-------------------------------------------------------|
| `%s`         | the test file                                         |
| `%S`         | the test's source directory                           |
| `%rules_dsl` | the runtime DSL shipped in the repo root (`rules.dsl`)|
| `mlir-opt-omp`, `mlir-opt`, `FileCheck` | resolved to the built/installed tools |

Tip: `CHECK-SAME` is used because `omp_lower.construct` prints its `runtime`,
`construct`, and the `pre`/`invoke`/`post` plans on a single line.

To capture the real output of a new pipeline while authoring CHECK lines:

```sh
mlir-opt-omp input.mlir --omp-lower-dsl=../rules.dsl --omp-lower-runtime=iomp \
  --omp-to-omp-lower --omp-outline --omp-lower-plan
```

### Making sure the test can fail

A test only pays for itself if it goes red when the behaviour it names breaks.
Two ways of getting that wrong have already slipped through here, both of which
left the suite green while nothing was being checked.

**Watch which stage the pipeline stops at.** Naming a runtime function is not
the same as checking it gets emitted. A test that stops at `--omp-to-omp-lower`
and matches `ext_pi_cl_team_fork` is reading it out of the *plan attribute* —
which says what the rules decided, not what came out. Moving that emission to a
different pass broke it without a single test noticing, because the only ones
naming the symbol looked at it a stage too early. When you change how something
is emitted, check that a test runs far enough down the pipeline to see it.

**A test with only `CHECK-NOT` asserts nothing.** "No `ident_t` is emitted for
libgomp" also holds for an empty module, so such a test cannot tell a correct
absence from a total failure to lower. Anchor it with a positive check that the
lowering did happen. The two often need separate `--check-prefix` runs: a
`CHECK-NOT` only covers up to the first positive match, and globals print above
the functions, so folding them into one prefix silently shrinks what the
negative check guards.

The quick self-test when writing one: *if I deleted the feature, would this
test fail?* If the answer needs thinking about, the test is not pinning it.

## Integration tests (`Integration/`)

End-to-end pipeline (C → CIR → MLIR → LLVM IR → object → linked binary) for each
runtime. Three drivers share one compile pipeline (`Integration/lib/common.sh`):

- `run_correctness.sh` compiles every PolyBench kernel with both a stock OpenMP
  compiler and the `mlir-opt-omp` pipeline and diffs the array dumps, validating
  that generated code runs and produces correct results.
- `run_performance.sh` times our tool against the native compiler (a 2×2
  seq/par × native/opt matrix) and reports per-kernel and geomean speedups.
- `tasks/run_tasks.sh` smoke-tests the `omp.task` lowering (libgomp): a hand-written
  MLIR case and a C case, both run and diffed against a reference.

Fully parametrized via env vars / `run.env` — see [`Integration/README.md`](Integration/README.md).

These depend on the full toolchain (clang/clangir, cir-opt, mlir tools, the
OpenMP runtime libraries) and a PolyBench checkout, so they are slower and
environment-dependent; keep them for whole-pipeline / numerical validation.

For a quick one-off check of the pipeline on a small C file — outside either
suite — see the per-runtime `compile-*.sh` scripts in
[`quick-compile/`](../quick-compile/).