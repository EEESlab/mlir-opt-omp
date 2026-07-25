# `test/demo` — one construct, whole pipeline, all runtimes (slide material)

A single OpenMP kernel taken **from C all the way down**, dumping every stage so
each slide can show one step of the lowering — the ClangIR front end included.

```
vecadd.c   #pragma omp parallel for
  │  clang -fclangir -emit-cir          →  out/00-frontend.cir        ClangIR (CIR dialect)
  │  cir-opt --cir-to-llvm              →  out/01-cir-to-llvm.mlir    omp + llvm dialects   ← input to our tool
  ├─ iomp / libgomp / pmsis ────────────────────────────────────────────────────────────
  │    mlir-opt-omp --omp-to-omp-lower  →  out/<rt>/02-omp-to-omp-lower.mlir   omp_lower.construct (plan as attrs)
  │                 --omp-outline       →  out/<rt>/03-omp-outline.mlir        region outlined → func.func, captures packed
  │                 --omp-lower-plan    →  out/<rt>/04-omp-lower-plan.mlir     concrete __kmpc_* / GOMP_* / ext_pi_* calls
  │    mlir-opt | mlir-translate | opt | llc | clang  →  out/<rt>/vecadd  → run   (iomp, libgomp)
       (pmsis stops at 04: the riscv32 / gvsoc back end needs the PULP SDK)
```

## Files

| file | role |
|------|------|
| `vecadd.c`    | the kernel shown on the slides — `#pragma omp parallel for` (fork + work-sharing loop in one) |
| `driver.c`    | `main()` that calls the kernel and prints the result (stock clang, linked in) |
| `run-demo.sh` | runs the whole pipeline, dumping every stage to `out/` |
| `out/`        | generated (gitignore-able); `00`/`01` shared, `02`–`04` per runtime |

## Run it

Tool paths come from `../Integration/config.env` (same file the Integration
drivers use). Copy an example once, then run:

```bash
cp ../Integration/config.env.lucap-wsl.example ../Integration/config.env
./run-demo.sh                 # all three runtimes
./run-demo.sh iomp            # just one
```

Native runtimes (`iomp`, `libgomp`) also build and **run** the binary and print
`0 11 22 33 44 55 66 77` — a self-check that the lowering is correct end to end.

## What each stage shows (map to slides)

- **`vecadd.c`** — the source. "This is all the user writes."
- **`00-frontend.cir`** — ClangIR's output: the OpenMP construct in the **CIR dialect**. Proves the front end is real, not hand-written MLIR.
- **`01-cir-to-llvm.mlir`** — after `cir-to-llvm`: `omp.parallel` / `omp.wsloop` in the `omp` + `llvm` dialects. **This is where our tool takes over.**
- **`02-omp-to-omp-lower.mlir`** — `omp.*` replaced by a single **`omp_lower.construct`** carrying the runtime-specific plan (`pre` / `invoke` / `post`) as MLIR attributes. **The DSL is read here.** Diff this file across runtimes to see the plan change.
- **`03-omp-outline.mlir`** — the region is outlined into `func.func @outlined_parallel_*`, captures packed per `capture_strategy`; the wsloop nest is lowered. The C++ **mechanism**.
- **`04-omp-lower-plan.mlir`** — the plan attributes become **concrete runtime calls**: `__kmpc_fork_call` (iomp) / `GOMP_parallel` (libgomp) / `ext_pi_cl_team_fork` (pmsis).

## Live-demo tip: the retargeting story is one flag

```bash
diff <(mlir-opt-omp --omp-lower-dsl=../../rules.dsl --omp-lower-runtime=iomp \
         --omp-to-omp-lower --omp-outline --omp-lower-plan out/01-cir-to-llvm.mlir) \
     <(mlir-opt-omp --omp-lower-dsl=../../rules.dsl --omp-lower-runtime=libgomp \
         --omp-to-omp-lower --omp-outline --omp-lower-plan out/01-cir-to-llvm.mlir)
```

Same input, one flag changed — **the diff is the point of the tool.**
