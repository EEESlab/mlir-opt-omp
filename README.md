# mlir-opt-omp

A custom [`mlir-opt`](https://mlir.llvm.org/docs/Tutorials/MlirOpt/) that lowers
the MLIR **OpenMP dialect** (`omp.*`) into calls to an OpenMP runtime library,
driven by a **declarative rule file**, [`rules.dsl`](rules.dsl), describing the
ABI of each runtime: which function to call, in what order, with which
arguments, and how captured variables are passed.

Adding a runtime, or changing how a construct is emitted, means editing that
file rather than the compiler. That makes the tool practical for non-standard
runtimes, embedded ones in particular: alongside the two host runtimes it lowers
OpenMP onto the **PMSIS** cluster API of PULP/GAP8 microcontrollers.

Input is MLIR holding `omp.*` operations in the `llvm` dialect, as produced from C
by the ClangIR front-end (`clang -fclangir -emit-cir` → `cir-opt --cir-to-llvm`).
That is the path this repository is tested around, but nothing
in the passes is tied to it: MLIR works as well, and so does any
other front-end that gets there (Flang after FIR→LLVM, for instance). Output is
MLIR where the OpenMP constructs have become plain `func.call` / `llvm.call`,
ready for `mlir-translate` and the LLVM back-end.

## Supported runtimes and constructs

| Construct | `iomp` (Intel/LLVM `__kmpc_*`) | `libgomp` (GCC `GOMP_*`) | `pmsis` (PULP cluster) |
|---|---|---|---|
| `omp.parallel` | ✅ `__kmpc_fork_call` | ✅ `GOMP_parallel` | ✅ `ext_pi_cl_team_fork` |
| `omp.barrier` | ✅ `__kmpc_barrier` | ✅ `GOMP_barrier` | ✅ `ext_pi_cl_team_barrier` |
| `omp.wsloop` | ✅ static — `__kmpc_for_static_init_4`<br>✅ dynamic — `__kmpc_dispatch_init_4`/`_next_4` | ✅ static — computed thread bounds<br>✅ dynamic — `GOMP_loop_dynamic_start`/`_next` | ✅ static — computed thread bounds |
| `omp.task` | ⏳ planned | ✅ `GOMP_task` | ⏳ API to be defined |


## Build instructions

### Requirements

- **CMake** ≥ 3.20 and a **C++17** compiler. The commands below pass
  `-G Ninja`, the usual choice for LLVM, but any generator works.
- An **LLVM/MLIR 23** build. See Step 1 for more details.
- The end-to-end tests also need a standard OpenMP compiler (clang for `iomp`,
  gcc for `libgomp`) and, for `pmsis`, the GAP SDK with the **gvsoc** simulator.

### 1. Get an LLVM/MLIR build

Which one you need depends on whether you want to compile **from C** or to use a `.mlir` file as input.

#### Option 1: With ClangIR for lowering from C

ClangIR is the front-end that turns C into the MLIR this tool consumes, and the
OpenMP part of it is not upstream yet: upstream ClangIR emits `omp.parallel`
and `omp.barrier`, while `#pragma omp for`, `parallel for` and `task` still hit
`errorNYI`. The emission this pipeline needs lives in the **EEESlab fork**.

```sh
git clone -b users/lucap/cir-omp-clauses \
  https://github.com/EEESlab/llvm-project.git
git -C llvm-project checkout 9be70f177d9b     # last known-good, see below
```

and build it with CIR enabled — the paths below are relative to the directory
holding the clone:

```sh
cmake -S llvm-project/llvm -B llvm-project/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" -DCLANG_ENABLE_CIR=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
  -DLLVM_DEFAULT_TARGET_TRIPLE="x86_64-unknown-linux-gnu" \
  -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_BUILD_TESTS=ON -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON \
  -DMLIR_BUILD_MLIR_C_DYLIB=ON
cmake --build llvm-project/build
```

Notes:

- `9be70f177d9b` is the commit the C pipelines below are tested against
  (`clang --version` then reports `23.0.0git … EEESlab/llvm-project`). Skip the
  `git checkout` to take the branch tip instead.
- `RISCV` in `LLVM_TARGETS_TO_BUILD` is only needed for the PULP/`pmsis` target.
- `LLVM_INSTALL_UTILS=ON` installs `FileCheck`, for Regression tests.
- No install step is needed: step 2 works against this build tree as it is. Add
  `-DCMAKE_INSTALL_PREFIX=<prefix>` above if you do want to install it.

#### Option 2: MLIR input only

Any LLVM/MLIR 23 build or install works — the passes run on the `omp` and `llvm`
dialects.

An install has to have been configured with `-DLLVM_INSTALL_UTILS=ON`, though:
without it there is no `FileCheck`, and the regression suite cannot run.


### 2. Build mlir-opt-omp

```sh
cmake -S . -B build -G Ninja -DMLIR_DIR=<LLVM_DIR>/lib/cmake/mlir
cmake --build build
```

`<LLVM_DIR>` is the **build tree** from step 1 — `<…>/llvm-project/build` — or an
LLVM **install** prefix, if you installed it. Everything resolves through
`LLVM_LIBRARY_DIR`, so both work unchanged.

Keep the build directory named `build`: it is what `local.env.example` and the
setups in [`docs/setups/`](docs/setups/) use for `OMP_TOOL_BIN`.

If `<LLVM_DIR>` is an install prefix and its build tree is gone, CMake warns
here that it found no `lit` and the regression suite will not run. Either
install one (`pipx install lit`, or `pip install --user lit`) — any `lit` on
`PATH` is picked up automatically — or add
`-DLLVM_EXTERNAL_LIT=<llvm-build-tree>/bin/llvm-lit` to the line above. Both are
read at configure time, so re-run `cmake -S . -B build` after either.


## Quick start

With the tool built, this is the shortest path to seeing it work. It needs
neither ClangIR nor PolyBench, and only the last step needs an OpenMP runtime.

**1. Run the regression suite.** No compiler or OpenMP runtime involved, so it
works anywhere the tool builds:

```sh
cmake --build build --target check-omp
```

**2. Watch the rule file decide the output.** The same module, lowered for two
different runtimes.

```sh
build/mlir-opt-omp test/Regression/wsloop/schedule-static/schedule-static-iomp.mlir \
  --omp-lower-dsl=rules.dsl \
  --omp-to-omp-lower --omp-outline --omp-lower-plan \
  --omp-lower-runtime=iomp | grep 'call @'
```

```mlir
call @__kmpc_for_static_init_4(...)
call @__kmpc_for_static_fini(...)
call @__kmpc_barrier(...)
llvm.call @sink(...)
llvm.call @__kmpc_fork_call(...)
```

Change one word — `--omp-lower-runtime=libgomp` — and the same loop comes out
against the GCC runtime instead:

```mlir
%15 = call @omp_get_thread_num() : () -> i32
%16 = call @omp_get_num_threads() : () -> i32
call @GOMP_barrier() : () -> ()
llvm.call @sink(...)
call @GOMP_parallel(...)
```

`@sink` is the loop body of that test case — everything around it is what the
rule file put there.

**3. Run it end to end.** The suite above checks IR. The integration drivers
take a real PolyBench kernel through clang→CIR→`mlir-opt-omp`→binary and compare
the numbers it prints against the same kernel built on the native runtime. They
need the full toolchain, which they locate through `local.env`:

```sh
cp local.env.example local.env && $EDITOR local.env   # LLVM_BIN, OMP_TOOL_BIN, INC_OMP
cd test/Integration && ./run_correctness.sh
```

From here: [Usage](#usage) for the pass pipeline, and
[`test/Integration/`](test/Integration/README.md) for the PolyBench
correctness and performance suites.

## Usage

Beyond the standard `mlir-opt` options, the tool adds two flags:

| Flag | Default | Meaning |
|---|---|---|
| `--omp-lower-dsl=<file>` | `rules.dsl` | the rule file describing the runtimes |
| `--omp-lower-runtime=<name>` | `iomp` | the `runtime` to target with the lowering |


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
| `--omp-barrier-elim` | optional, runs first: drops redundant team barriers. It doesn't read DSL |
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

A runnable version of this, per runtime, is what the integration drivers do —
see [`test/Integration/`](test/Integration/README.md).

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

`local.env` holds the per-machine tool paths and is git-ignored. Ready-made ones
for the machines this project is developed on are in
[`docs/setups/`](docs/setups/). What the
tests *run* is separate and optional
([`test/Integration/run.env.example`](test/Integration/run.env.example)).

See [`test/Integration/README.md`](test/Integration/README.md) for the full
configuration reference, the PULP/gvsoc target, and the reported metrics.

## Repository layout

Headers under `include/OmpLowering/`, implementations under `lib/`, the tool
under `tools/` — the layout of
[`mlir/examples/standalone`](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone),
the out-of-tree MLIR template this project follows. Headers are included by full
path (`#include "OmpLowering/IR/OmpLoweringOps.h"`), and the generated `.inc`
files mirror that same path inside the build tree.

```
rules.dsl                    the lowering rules for the three runtimes
include/OmpLowering/
  IR/                        omp_lower dialect: OmpLoweringOps.{td,h}
  DSL/                       DSLParser.h, DSLEvaluator.h
  Transforms/                the three pass headers
lib/
  DSL/                       lexer/parser + evaluator — no MLIR dependency
  IR/                        dialect implementation
  Transforms/                the three passes
tools/mlir-opt-omp/          the executable: flag extraction, registration
test/                        regression + integration suites
docs/
  lowering-specs/            lowering specifications (ident_t parity, omp.task)
                             and DSL design notes
  setups/                    ready-made local.env files, per dev machine
scripts/                     load-local-env.sh — the tool-path resolution the
                             test drivers share
```

The sources behind it:

| File | Does |
|---|---|
| `lib/DSL/DSLParser.cpp` | lexer/parser producing the rule AST |
| `lib/DSL/DSLEvaluator.cpp` | evaluates rules against an op, producing a lowering plan |
| `lib/Transforms/OmpToOmpLowerPass.cpp` | pass 1 — `omp.*` → `omp_lower.construct` |
| `lib/Transforms/OmpOutliningPass.cpp` | pass 2 — outlining and `wsloop` lowering |
| `lib/Transforms/PlanLoweringPass.cpp` | pass 3 — plans → runtime calls |
| `include/OmpLowering/IR/OmpLoweringOps.td` | TableGen definition of the `omp_lower` dialect |
| `tools/mlir-opt-omp/mlir-opt-omp.cpp` | flag extraction, dialect/pass registration |

They build into three libraries: `OmpLoweringDSL` (rule file front end, links
only `LLVMSupport`), `MLIROmpLowering` (the dialect) and
`MLIROmpLoweringTransforms` (the passes, where the two meet).

## Extending

To add a construct or a runtime, start from [`rules.dsl`](rules.dsl): each
`runtime` block declares its constructs, and each construct its
`capture_strategy` — the sole ABI selector, which also fixes the outlined
function's signature — and its blocks. A block is any name followed by `{ ... }`
and says *when* its calls run: `pre`, `invoke` and `post` around the construct,
and for a work-sharing loop whose iterations the runtime hands out a chunk at a
time, `first_chunk` and `next_chunk` around each chunk. `next_chunk` is what
makes a loop chunked; `first_chunk` is only for a runtime whose opening call
differs from the repeat one, as libgomp's does.

Such a loop also has three properties for the shape of its dispatch ABI. Each
defaults to what iomp's `__kmpc_dispatch_*_4` does, so a runtime that agrees
declares none of them, and a value none of them understands is an error rather
than a silent fallback:

| property | default | says |
|---|---|---|
| `chunk_index` | the induction variable's type | width of the bound slots and of the values passed by value (`i64` for libgomp, which is `long`-based whatever the loop is) |
| `chunk_result` | `i32` | what the acquisition call answers with (`i8` for a C `_Bool`) |
| `chunk_bound` | `inclusive` | whether the upper bound written into the slot is the last valid iteration or the one past it |

A
change there is picked up at run time — no rebuild needed — which makes it easy
to iterate with a regression test. Constructs whose lowering needs more than the
DSL expresses (new `emit` primitives, a different outline shape) also require
C++ work in `lib/Transforms/OmpOutliningPass.cpp`.

Every new feature should come with a regression test under `test/Regression/`;
[`test/README.md`](test/README.md) documents the format and the available lit
substitutions.
