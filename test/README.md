# Tests

Two layers of tests, following the LLVM/MLIR convention, plus the
lines-of-code measurement the paper reports.

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
`barrier-elim/` is the exception: it covers a pass rather than a construct, so
it is named after the pass.

```
Regression/
  parallel/   if/  num_threads/  proc_bind/  private/  firstprivate/
  wsloop/     schedule-static/  schedule-dynamic/  schedule-guided/
              collapse/  nowait/
  barrier/
  task/       if/  firstprivate/
  taskwait/
  barrier-elim/
```

Most `barrier-elim/` tests select no runtime at all — `--omp-barrier-elim`
reads no DSL — so one RUN line covers all three. The `no-barrier-call-*.mlir`
trio is the exception, and carries the result through to the emitted runtime
call.

lit recurses, so a test is picked up wherever it lands. **Every clause in the
matrix below has a directory with at least one test per runtime.** A new clause
gets its directory as soon as the lowering accepts it — if there is nothing to
assert yet beyond "this is dropped on the floor", that is itself the test to
write (see *Encoding a gap*). An empty clause directory is a gap to fill, not a
directory to delete; git does not track empty ones, so it needs a `.gitkeep`.

### Coverage

Clauses supported by the lowering, and whether a regression test covers them.
Every cell now has a test; the symbol says what that test asserts.

| Construct | Clause | iomp | libgomp | pmsis |
|---|---|:--:|:--:|:--:|
| `parallel` | — | ✓ | ✓ | ✓ |
| | `if` | ✓ | ✓ | ✓ |
| | `num_threads` | ✓ | ✓ | ✓ |
| | `proc_bind` | ✓ | ✓ | ! |
| | `private` | ✓ | ✓ | ✓ |
| | `firstprivate` | ✓ | ✓ | ✓ |
| `wsloop` | — | ✓ | ✓ | ✓ |
| | `schedule(static)` | ✓ | ✓ | ✓ |
| | `schedule(static, N)` | ✗ | ✗ | ✗ |
| | `schedule(dynamic)` | ✓ | ✓ | ! |
| | `schedule(dynamic, N)` | ✓ | ✓ | ! |
| | `schedule(guided)` | ! | ! | ! |
| | `collapse(N)` | ! | ! | ! |
| | `nowait` | ✓ | ✓ | ✓ |
| `barrier` | — | ✓ | ✓ | ✓ |
| `task` | — | ✓ | ✓ | n/a |
| | `if` | ✓ | ✓ | n/a |
| | `firstprivate` | ✓ | ✓ | n/a |
| `taskwait` | — | ✓ | ✓ | n/a |

`✓` a passing test covers the lowering, `n/a` the runtime has no such construct
(`rules.dsl` declares no `task`/`taskwait` for pmsis), `!` the clause is *not*
supported and a passing test asserts the diagnostic rather than a lowering,
`✗` not supported **and not diagnosed** — the clause is accepted and silently
dropped or mislowered, with an `XFAIL`ed test stating what it should do. Every
`✗` is a latent wrong-code path:

- **`schedule(static, N)`** matches `construct wsloop when schedule == static`
  on every runtime and the chunk size is then dropped. libgomp and pmsis run
  `emit thread_bounds`, one contiguous block per thread, which is not the
  round-robin distribution of `N` iterations the clause asks for; iomp passes
  schedule constant 34 with `default_chunk` rather than 33 with the clause's
  value. No test covers it yet — the `✗` is the record that it is known.

Three more gaps are properties of the loop rather than of a clause, so they have
no row of their own:

- **A descending loop** — a negative step — runs zero iterations under every
  wsloop lowering. The comparison is fixed at `sle` (iomp) or `slt` (the two
  inline ones, and libgomp's dynamic chunks), all of which are false on entry
  when the loop counts down. The trip count the iomp bound is derived from is
  wrong for a negative step too. Which way to fix it is open: a step known to be
  negative could be rejected outright, the way pmsis rejects `schedule(dynamic)`
  today, while a step whose sign is only known at run time would need the
  comparison chosen there.
- **A non-`i32` induction variable** — `__kmpc_for_static_init_4` and
  `__kmpc_dispatch_*_4` are the 32-bit entry points, and nothing checks that the
  loop agrees with them. An `i64` loop gets `i64` bound slots, of which those
  entry points write only the low half, and the schedule constant is widened to
  match. The real answer is the `_8` variants, guarded on the loop's own width.
  libgomp is unaffected: its `chunk_index = i64` makes the conversion explicit
  at the ABI boundary.
- **The `nonmonotonic` modifier** — since OpenMP 5.0 `schedule(dynamic)` is
  nonmonotonic by default and clang emits `35 | (1<<30)`. `rules.dsl` emits a
  plain 35, which asks the runtime for the ordered dispatch path rather than the
  work-stealing one: slower, not wrong. An explicit `schedule(nonmonotonic:
  dynamic)` is dropped with no diagnostic — the pass seeds no `schedule_mod` in
  the evaluation context, so the rules have no way to name it.

### Encoding a gap

A green suite must not imply a feature exists, so unimplemented behaviour is
recorded one of two ways depending on what the tool does today:

- **Rejected cleanly** — name the test `*-unsupported-*.mlir` and assert the
  diagnostic with `-verify-diagnostics`. It passes, and what it locks in is a
  real property: the tool refuses the input instead of mislowering it. Grep for
  `-unsupported-` to list these.
- **Silently wrong** — write the test against the behaviour the tool *should*
  have and mark it `// XFAIL: *`. It fails, and `check-omp` runs lit with
  `--show-xfail`, so every run prints the gap under `Expectedly Failed Tests`
  **by name**. When the fix lands the test XPASSes, which lit reports as a
  failure — that is the prompt to drop the `XFAIL` line.

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
- `constructs/run_constructs.sh` covers the constructs and clauses PolyBench
  never writes — which is most of the matrix: across the whole suite it uses
  only `parallel`, a bare `for` and `private`. One standalone C program per
  clause, each printing `42` only if the clause actually took effect.

Fully parametrized via env vars / `run.env` — see [`Integration/README.md`](Integration/README.md).

These depend on the full toolchain (clang/clangir, cir-opt, mlir tools, the
OpenMP runtime libraries) and a PolyBench checkout, so they are slower and
environment-dependent; keep them for whole-pipeline / numerical validation.

## Lowering complexity analysis (`LoC/`)

Not tests: the two scripts behind the lines-of-code comparison of Section 4.4,
which count how much of GCC and of Clang a minimal lowering of the supported
constructs takes. They run over external GCC and llvm-project checkouts, at the
commits the paper pins, and reproduce Tables 1 and 2 of the supplementary
material. See [`LoC/README.md`](LoC/README.md).
