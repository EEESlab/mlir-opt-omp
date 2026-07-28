# mlir-opt-omp

A custom [`mlir-opt`](https://mlir.llvm.org/docs/Tutorials/MlirOpt/) that lowers
the MLIR **OpenMP dialect** (`omp.*`) into calls to an OpenMP runtime
library, driven by a **declarative rule file**, called [`rules.dsl`](rules.dsl) describing the ABI of the supported runtimes (which function to call, in what order, with which
arguments, how captured variables are passed).

Adding support for a new runtime, or changing how an
existing construct is emitted, normally means editing that file, not the
compiler. This makes the tool useful for targeting non-standard runtimes, in
particular embedded ones: alongside the two host runtimes it can lower OpenMP
onto the **PMSIS** cluster API of PULP/GAP8 microcontrollers.

Input is MLIR holding `omp.*` operations over memory already in the `llvm`
dialect — captured variables are recognised through `llvm.alloca`. That is the
state reached from C by the ClangIR front-end (`clang -fclangir -emit-cir` →
`cir-opt --cir-to-llvm`), which is the path this repository is built and tested
around, but nothing in the passes is tied to it: hand-written MLIR works just
as well, and so does any other front-end that gets there (Flang after the
FIR→LLVM conversion, for instance). Output is MLIR in which the OpenMP
constructs have become plain `func.call` / `llvm.call` operations, ready for
`mlir-translate` and the usual LLVM back-end.

## Supported runtimes and constructs

| Construct | `iomp` (Intel/LLVM `__kmpc_*`) | `libgomp` (GCC `GOMP_*`) | `pmsis` (PULP cluster) |
|---|---|---|---|
| `omp.parallel` | ✅ `__kmpc_fork_call` | ✅ `GOMP_parallel` | ✅ `ext_pi_cl_team_fork` |
| `omp.barrier` | ✅ `__kmpc_barrier` | ✅ `GOMP_barrier` | ✅ `ext_pi_cl_team_barrier` |
| `omp.wsloop` | ✅ static — `__kmpc_for_static_init_4` | ✅ static — computed thread bounds | ✅ computed thread bounds |
| `omp.task` | ⏳ planned | ✅ `GOMP_task` | ⏳ API to be defined |

The two capture strategies differ per runtime: `iomp` passes captured variables
as individual pointer arguments to the microtask (`by_pointer`), while
`libgomp` and `pmsis` pack them into an environment struct passed as a single
closure argument (`packed`). See [`docs/`](docs/) for the detailed lowering
specifications.

## Requirements

- **CMake** ≥ 3.20 and a **C++17** compiler.
- An **LLVM/MLIR build**. ClangIR (`-DCLANG_ENABLE_CIR=ON`) is what you want if
  you compile from C, since it is the front-end of that pipeline; a stock
  LLVM/MLIR is enough to build and use the tool on MLIR input — see
  [Building without ClangIR](#building-without-clangir).
- To run the end-to-end tests, additionally: a traditional OpenMP compiler (clang for
  `iomp`, gcc for `libgomp`) and, for `pmsis`, the GAP SDK with the **gvsoc**
  simulator.

### Building the LLVM toolchain

```sh
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_INSTALL_PREFIX="<LLVM_INSTALL_DIR>" \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" -DCLANG_ENABLE_CIR=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
  -DLLVM_DEFAULT_TARGET_TRIPLE="x86_64-unknown-linux-gnu" \
  -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_BUILD_TESTS=ON -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON \
  -DMLIR_BUILD_MLIR_C_DYLIB=ON \
  <LLVM_SRC_DIR>/llvm-project/llvm
ninja && ninja install
```

`RISCV` in `LLVM_TARGETS_TO_BUILD` is only needed for the PULP/`pmsis` target.
`LLVM_INSTALL_UTILS=ON` matters for the test suite — it installs `FileCheck`
and `llvm-lit`.

## Building

```sh
cmake -S . -B build -G Ninja -DMLIR_DIR=<LLVM_DIR>/lib/cmake/mlir
cmake --build build
```

This produces `build/mlir-opt-omp`. Keep the directory named `build`: it is
what `local.env.example` and the setups in [`docs/setups/`](docs/setups/) use
for `OMP_TOOL_BIN`, and what the `quick-compile/` scripts fall back to.

`<LLVM_DIR>` can be either an LLVM **install** prefix or an LLVM **build
tree** — the latter is the common case with ClangIR, which is often built but
never installed. Everything is resolved through `LLVM_LIBRARY_DIR`, so both
work without further configuration.

### Building without ClangIR

The passes work on the `omp` and `llvm` dialects and never inspect a `cir.*`
operation; the CIR dialect is registered only so modules straight from the C
front-end still parse. CMake looks for `libMLIRCIR.a` in `LLVM_LIBRARY_DIR` and
links CIR when it is there, so the configure line above needs no change either
way. The configure output says which way it went:

```
-- mlir-opt-omp: CIR support ON
```

Force it either way with `-DOMP_LOWER_ENABLE_CIR=ON|OFF`:

```sh
cmake -S . -B build -G Ninja -DMLIR_DIR=<LLVM_DIR>/lib/cmake/mlir \
  -DOMP_LOWER_ENABLE_CIR=OFF
```

> Auto-detection only picks the **default**. CMake caches the option, so a
> directory first configured when CIR was not detectable keeps it off even
> after the LLVM install is fixed. Reconfigure with an explicit
> `-DOMP_LOWER_ENABLE_CIR=ON`, or configure into a clean directory.

Such a build lowers hand-written or Flang-produced MLIR exactly like the full
one. To try it: `quick-compile/compile-from-mlir.sh` takes a small OpenMP kernel
written directly in the `omp` + `llvm` dialects all the way to a running
binary and checks the result, and the regression suite (`ninja check-omp`)
passes unchanged — none of its tests involve CIR. What it cannot do is read a
module that still carries
`cir.*` attributes: unregistered dialect attributes do not parse, and
`--allow-unregistered-dialect` does not help. Modules from `cir-opt
--cir-to-llvm` do keep a few, which is why the sample pipelines strip them with
`sed` (see [`quick-compile/compile-iomp.sh`](quick-compile/compile-iomp.sh)).

## Usage

Beyond the standard `mlir-opt` options, the tool adds two flags:

| Flag | Default | Meaning |
|---|---|---|
| `--omp-lower-dsl=<file>` | `rules.dsl` | the rule file describing the runtimes |
| `--omp-lower-runtime=<name>` | `iomp` | which `runtime` block of that file to use |

Both are consumed before `MlirOptMain` parses the command line, so they must be
passed as `--flag=value` (not as two separate arguments).

The lowering runs as three passes, in this order:

```
omp.*  ──[--omp-to-omp-lower]──▶  omp_lower.construct  ──[--omp-outline]──▶  outlined func.func
                                                                                     │
                                                                       [--omp-lower-plan]
                                                                                     ▼
                                                                        func.call / llvm.call
```

| Pass | Does |
|---|---|
| `--omp-to-omp-lower` | reads the DSL and turns each `omp.*` op into an `omp_lower.construct` carrying its `pre`/`invoke`/`post` plan |
| `--omp-outline` | outlines construct body regions into `func.func` with the runtime's entry signature, and lowers `omp.wsloop` |
| `--omp-lower-plan` | replaces each `omp_lower.construct` with the concrete runtime calls |

Typical invocation:

```sh
mlir-opt-omp input.mlir \
  --allow-unregistered-dialect \
  --omp-lower-dsl=rules.dsl \
  --omp-lower-runtime=iomp \
  --omp-to-omp-lower --omp-outline --omp-lower-plan \
  -o lowered.mlir
```

### End-to-end, from C

The full pipeline — C to a linked binary — is:

```sh
# C -> CIR -> MLIR
clang -S -Xclang -fclangir -Xclang -emit-cir -fopenmp -I$INC_OMP test.c -o test.cir
cir-opt test.cir --cir-to-llvm --reconcile-unrealized-casts -o s1.mlir
sed -i -E 's/cir\.[^,}]+,? ?//g' s1.mlir          # drop leftover cir.* attributes

# OpenMP lowering (this tool)
mlir-opt-omp s1.mlir --allow-unregistered-dialect \
  --omp-lower-dsl=rules.dsl --omp-lower-runtime=libgomp \
  --omp-to-omp-lower --omp-outline --omp-lower-plan > s2.mlir

# MLIR -> LLVM IR -> object -> binary
mlir-opt s2.mlir --convert-arith-to-llvm --convert-func-to-llvm \
  --reconcile-unrealized-casts -o s3.mlir
mlir-translate s3.mlir --mlir-to-llvmir > s4.ll
llc -relocation-model=pic -filetype=obj s4.ll -o test.o
clang -O3 -fopenmp test.o main.o -o test
```

Runnable versions of this, per runtime, are in
[`quick-compile/`](quick-compile/README.md) — the fastest way to try the tool on
a small kernel.

## Testing

Two layers, described in [`test/README.md`](test/README.md).

**Regression tests** (lit + FileCheck) exercise the passes on hand-written IR.
No C compiler or OpenMP runtime involved, so they are fast and are the place to
lock in behaviour for every new feature:

```sh
cmake --build build --target check-omp
```

**Integration tests** run the whole pipeline on PolyBench kernels and check the
results against a stock OpenMP compiler, both for correctness (bit-identical
array dumps) and performance (speedup vs. native). They need the full toolchain
and a per-machine config:

```sh
cp local.env.example local.env && $EDITOR local.env   # LLVM_BIN, OMP_TOOL_BIN, INC_OMP
cd test/Integration
./run_correctness.sh                     # bundled kernels, iomp
RUNTIME=libgomp ./run_performance.sh
```

`local.env` holds the per-machine tool paths and is git-ignored; the
`quick-compile/` scripts read the same file. Ready-made ones for the machines
this project is developed on are in [`docs/setups/`](docs/setups/). What the
tests *run* is separate and optional
([`test/Integration/run.env.example`](test/Integration/run.env.example)).

See [`test/Integration/README.md`](test/Integration/README.md) for the full
configuration reference, the PULP/gvsoc target, and the reported metrics.

## Repository layout

| Path | Contents |
|---|---|
| [`rules.dsl`](rules.dsl) | the lowering rules for the three runtimes |
| `DSLParser.{h,cpp}` | lexer/parser producing the rule AST |
| `DSLEvaluator.{h,cpp}` | evaluates rules against an op, producing a lowering plan |
| `OmpToOmpLowerPass.cpp` | pass 1 — `omp.*` → `omp_lower.construct` |
| `OmpOutliningPass.cpp` | pass 2 — outlining and `wsloop` lowering |
| `PlanLoweringPass.cpp` | pass 3 — plans → runtime calls |
| `OmpLoweringOps.td` | TableGen definition of the `omp_lower` dialect |
| `mlir-opt-omp.cpp` | the executable: flag extraction, dialect/pass registration |
| [`docs/`](docs/) | lowering specifications (`ident_t` parity, `omp.task`) |
| [`test/`](test/README.md) | regression + integration test suites |
| [`quick-compile/`](quick-compile/README.md) | one-shot pipeline scripts for a small kernel |

## Extending

To add a construct or a runtime, start from [`rules.dsl`](rules.dsl): each
`runtime` block declares its constructs, and each construct its
`outline_signature`, `capture_strategy` and `pre`/`invoke`/`post` blocks. A
change there is picked up at run time — no rebuild needed — which makes it easy
to iterate with a regression test. Constructs whose lowering needs more than the
DSL expresses (new `emit` primitives, a different outline shape) also require
C++ work in `OmpOutliningPass.cpp`.

Every new feature should come with a regression test under `test/Regression/`;
[`test/README.md`](test/README.md) documents the format and the available lit
substitutions.
