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

### Running

From the build directory:

```sh
cmake --build . --target check-omp
```

### Adding a test

Create `Regression/<name>.mlir`:

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