# Specification: Clang-parity `ident_t` lowering for the iomp path

Scope: iomp (`__kmpc_*`) lowering only. 

Decisions: **flags are DSL-driven**; **psource uses the default placeholder string** (`;unknown;unknown;0;0;;`), with real source-location formatting deferred as a later development.

---

## 1. Goal

Make the `%struct.ident_t` values match what Clang/`OMPIRBuilder` produce.

1. **Per-construct `flags`** — barriers and worksharing-loop calls must carry the correct `OpenMPLocationFlags` bits, not a fixed `KMPC`.
2. **Non-null `psource` + correct `reserved_3` length** — every ident must point at a real source-location string (placeholder for now), with `reserved_3 = strlen`.

---

## 2. Clang reference

Pulled from the local LLVM checkout, these are the values we want to match.

### 3.1 Flag bits — `OpenMPLocationFlags`

```
OMP_IDENT_IMD                  = 0x01
OMP_IDENT_KMPC                 = 0x02   // always OR'd in by getOrCreateIdent
OMP_ATOMIC_REDUCE              = 0x10
OMP_IDENT_BARRIER_EXPL         = 0x20
OMP_IDENT_BARRIER_IMPL         = 0x40
OMP_IDENT_BARRIER_IMPL_FOR     = 0x40
OMP_IDENT_BARRIER_IMPL_SECTIONS= 0xC0
OMP_IDENT_BARRIER_IMPL_SINGLE  = 0x140
OMP_IDENT_WORK_LOOP            = 0x200
OMP_IDENT_WORK_SECTIONS        = 0x400
OMP_IDENT_WORK_DISTRIBUTE      = 0x800
```

### 3.2 Which flag each construct uses

- **parallel** (`__kmpc_fork_call`, `__kmpc_push_num_threads`, `__kmpc_push_proc_bind`, `__kmpc_global_thread_num`): plain ident, `Flags = 0` → effective **`0x02`**. (Clang `emitUpdateLocation(CGF, Loc)` with default flags.)
- **explicit barrier** (`omp.barrier` → `__kmpc_barrier`): `getDefaultFlagsForBarriers(OMPD_barrier)` = `BARRIER_EXPL` → effective **`0x22`**.
- **wsloop `for_static_init` / `for_static_fini`**: `OMP_IDENT_WORK_LOOP` → effective **`0x202`**.
- **wsloop trailing implicit barrier** (`when not nowait => __kmpc_barrier`): this is the loop's *implicit* barrier, `getDefaultFlagsForBarriers(OMPD_for)` = `BARRIER_IMPL_FOR` → effective **`0x42`**. 

### 3.3 The `ident` initializer
fields `{0, flags, reserve2, SrcLocStrSize, SrcLocStr}`; global is **private, constant, `unnamed_addr`, `align 8`**. Deduplicated via `IdentMap` keyed on `{SrcLocStr, (LocFlags<<31)|Reserve2Flags}` plus a module-globals scan.

### 3.4 The `psource` string

- Format: `;file;function;line;col;;` (`getOrCreateSrcLocStr(fn,file,line,col)`).
- Default / no-debug-info fallback: **`;unknown;unknown;0;0;;`** (`getOrCreateDefaultSrcLocStr`), length **22** (null terminator not counted in `SrcLocStrSize`).
- Interned in `SrcLocStrMap`; one global string reused by all idents with that text.

`reserved_2` host default is `0` (`getDefaultLocationReserved2Flags`, base returns 0).

---

## 4. Design

### 4.1 DSL surface — `ident(<flag>)` argument form

The flag travels with each *call argument*, exactly like Clang attaches it per call site. 
`ident(work_loop)` parses as `CallExpr{name:"ident", args:[IdentExpr{"work_loop"}]}`.

Accepted forms in any call-argument position:

| DSL token            | effective flags |
|----------------------|-----------------|
| `ident` / `ident(kmpc)`        | `0x02` |
| `ident(barrier_expl)`          | `0x22` |
| `ident(barrier_impl)` / `ident(barrier_impl_for)` | `0x42` |
| `ident(barrier_impl_sections)` | `0xC2` |
| `ident(barrier_impl_single)`   | `0x142` |
| `ident(work_loop)`             | `0x202` |
| `ident(work_sections)`         | `0x402` |
| `ident(work_distribute)`       | `0x802` |


### 4.2 Evaluator change (`DSLEvaluator.cpp`)

`ident(...)` must not eval its arg through `scope.get` (the flag token isn't a scope variable and would error). 

The plan attribute carries a `StringAttr` like `"%ident:work_loop"`. The existing symbolic-string convention (`%ident`) is preserved; we just add an optional `:flag` suffix.

### 4.3 C++ lowering change (`OmpOutliningPass.cpp`)

Replace the bespoke single-global builder with a Clang-style interning helper:

```cpp
// flags already includes KMPC (0x02). Interns one global per distinct flags
// value; all share the single default psource string global.
static Value getOrCreateIdent(ModuleOp module, OpBuilder &b, Location loc,
                              MLIRContext *ctx, uint32_t flags);
```

Behaviour:
1. Ensure the shared psource global exists: private constant `i8` array
   `";unknown;unknown;0;0;;\00"` (23 bytes), name e.g. `__omp_src_loc_default`.
2. Ident global name keyed by flags, e.g. `__omp_ident_<flags-hex>` (`__omp_ident_2`, `__omp_ident_22`, `__omp_ident_202`, `__omp_ident_42`, …). `lookupSymbol` → reuse; else create.
3. Initializer fields: `{0, flags, 0, 22, addrof(__omp_src_loc_default)}`.
   Set the global `constant`, `Private`, and (to match Clang) `unnamed_addr` + `align 8`.
4. Return `AddressOfOp` of the ident global.

A small `flagTokenToBits(StringRef)` maps the DSL tokens in §4.1 to bits (always OR `0x02`); unknown token → pass error.

Resolution sites that currently match `"ident"`/`"%ident"` must parse the optional `:flag` suffix and call `getOrCreateIdent(..., bits)`:
- parallel pre/arg resolution, and the call-site `identVal` block (materialised on demand — the fork always needs an ident, so it is no longer gated by an `emit ident`);
- barrier arg resolution;
- wsloop `resolveCallArg`


### 4.4 Note on the thread-id ident

Clang uses a *plain* (`0x02`) ident for `__kmpc_global_thread_num` even inside a flagged barrier. In this tool the gtid inside outlined functions comes from the microtask arg or `omp_get_thread_num`, not from `__kmpc_global_thread_num(ident)`, so this nuance only touches the parallel call site — which already uses the default ident. No special handling required.

---

## 5. Worked example (iomp, explicit barrier)

Before:
```llvm
@__omp_ident_0 = private constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr null }
; ... __kmpc_barrier(ptr @__omp_ident_0, i32 %tid)
```

After:
```llvm
@__omp_src_loc_default = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
@__omp_ident_2  = private unnamed_addr constant %struct.ident_t { i32 0, i32 2,  i32 0, i32 22, ptr @__omp_src_loc_default }, align 8
@__omp_ident_22 = private unnamed_addr constant %struct.ident_t { i32 0, i32 34, i32 0, i32 22, ptr @__omp_src_loc_default }, align 8
; ... __kmpc_barrier(ptr @__omp_ident_22, i32 %tid)   ; 34 = 0x22 = KMPC|BARRIER_EXPL
```

---

## 6. Testing

- **DSL parse/eval unit**: `ident(work_loop)` → plan arg `"%ident:work_loop"`; bare `ident` → `"%ident"`. Unknown flag token → diagnostic.
- **IR FileCheck** (extend `test/`): for an iomp parallel+barrier+static-wsloop input, check the emitted globals' `flags` fields are `2` / `34` (0x22) / `514` (0x202) / `66` (0x42) and that `reserved_3 = 22`, `psource` = `@__omp_src_loc_default`.
- **Dedup**: two barriers / two static loops share one `__omp_ident_22` / `__omp_ident_202`.

---

## 7. Non-goals / deferred

- Real source-location `psource` (`;file;func;line;col;;` from MLIR `Location`). Layered later into `getOrCreateSrcLocStr` without touching ident/flags structure or the DSL surface.
- `sections` / `single` / `distribute` constructs — flag tokens are defined for completeness but those constructs aren't lowered yet.
- `OMP_ATOMIC_REDUCE`, cancellation-barrier (`__kmpc_cancel_barrier`) variants.

## 8. Implementation order

1. `getOrCreateIdent(module, flags)` + shared psource global + `flagTokenToBits`; delete inline builder and `getIdentAddr`. (No behaviour change yet: callers still request `0x02`.)
2. Evaluator: `ident(<flag>)` → `"%ident:<flag>"`; bare `ident` unchanged.
3. Pass: parse `:flag` suffix at the three resolution sites; route to `getOrCreateIdent`.
4. `rules.dsl`: add flag tokens to barrier + wsloop calls.
5. Tests (§6).
