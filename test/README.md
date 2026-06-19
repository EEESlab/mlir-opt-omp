# Tests

Two layers, following the LLVM/MLIR convention.

## Regression tests (`Regression/`) — lit + FileCheck

Fast, hermetic tests of the IR transforms. Each `.mlir` file embeds the command
to run (`// RUN:`) and the expected output (`// CHECK:`). No C compiler or
OpenMP runtime is involved — they exercise the `mlir-opt-omp` passes directly on
hand-written IR, so they run in milliseconds and are the right place to lock in
behaviour for every new feature.

This mirrors `mlir/examples/standalone/test/` in llvm-project, the canonical
out-of-tree MLIR test setup.

### Running

From the build directory:

```sh
ninja check-omp
```

or point lit at the source tree directly:

```sh
<llvm-build>/bin/llvm-lit -sv test/
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
runtime. `Integration/run_correctness.sh` compiles every PolyBench kernel with
both a stock OpenMP compiler and the `mlir-opt-omp` pipeline and diffs the array
dumps, so it validates that generated code actually runs and produces correct
results. Fully parametrized via env vars / `config.env` — see
[`Integration/README.md`](Integration/README.md).

These depend on the full toolchain (clang/clangir, cir-opt, mlir tools, the
OpenMP runtime libraries) and a PolyBench checkout, so they are slower and
environment-dependent; keep them for whole-pipeline / numerical validation.

> The legacy `compile-*.sh` scripts in this directory are earlier single-kernel
> prototypes, superseded by `Integration/run_correctness.sh`.
