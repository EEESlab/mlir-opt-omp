# Specification: `omp.task` lowering across runtimes

Status:

| Runtime | Status        |
|---------|---------------|
| libgomp | **implemented** (v2) |
| iomp    | **implemented** (v1: tied tasks; `if`/`final` and firstprivate deferred) |
| pmsis   | planned (phase 3, API to be defined) |

v1: **closure/packed reuse**, **DataLayout with fallback-16** for the env-struct alignment.
v2: **nested tasks** — tasks inside a `parallel` or inside another `task` — plus an end-to-end run-against-libgomp integration test.

---

## 1. Goal

Lower `omp.task` so that the captured environment is snapshotted and the task body runs through the selected runtime's task API, matching each runtime's ABI:

1. The task body is **outlined** into a function whose signature matches the runtime's task-entry convention.
2. The captured variables are packaged per the runtime's calling convention (a copied closure struct for libgomp, a `kmp_task_t` shareds block for iomp).
3. The `if` clause selects the deferred-vs-undeferred path where the runtime distinguishes them.

OpenMP semantics: variables referenced in a task are **firstprivate by default**.
The existing capture-packing already provides these snapshot semantics; explicit `private`/`firstprivate` recipes are handled via the shared firstprivate-injection step.

---

## 2. Runtime reference

### 2.1 libgomp — `GOMP_task`

```c
void GOMP_task (void (*fn)(void *data),
                void *data,
                void (*cpyfn)(void *dst, void *src),
                long  arg_size,
                long  arg_align,
                bool  if_clause,
                unsigned flags,
                void **depend,
                int   priority,
                void *detach);
```

Key behaviour: with `cpyfn == NULL`, libgomp allocates `arg_size` bytes aligned to `arg_align`, `memcpy`s `data` into that task-private block, and later calls `fn(copy)`. 
This is true even for undeferred tasks (`if_clause == false` / `final`). 
Therefore the outlined body is an ordinary closure `void (*)(void *)` and the only task-specific data is `arg_size` / `arg_align`.

ABI mapping used by the tool:

| `GOMP_task` param | C type           | LLVM type |
|-------------------|------------------|-----------|
| `fn`              | `void(*)(void*)` | `ptr`     |
| `data`            | `void*`          | `ptr`     |
| `cpyfn`           | `void(*)(…)`     | `ptr` (`null`) |
| `arg_size`        | `long`           | `i64`     |
| `arg_align`       | `long`           | `i64`     |
| `if_clause`       | `_Bool`          | `i8`      |
| `flags`           | `unsigned`       | `i32` (`0`) |
| `depend`          | `void**`         | `ptr` (`null`) |
| `priority`        | `int`            | `i32` (`0`) |
| `detach`          | `void*`          | `ptr` (`null`) |

### 2.2 iomp — `__kmpc_omp_task_alloc` + `__kmpc_omp_task`

```c
kmp_task_t *__kmpc_omp_task_alloc(ident_t *loc, int gtid, int flags,
                                  size_t sizeof_kmp_task_t,
                                  size_t sizeof_shareds,
                                  kmp_routine_entry_t task_entry);
int __kmpc_omp_task(ident_t *loc, int gtid, kmp_task_t *new_task);

// undeferred (if(0) / final) path:
void __kmpc_omp_task_begin_if0   (ident_t*, int gtid, kmp_task_t*);
void __kmpc_omp_task_complete_if0(ident_t*, int gtid, kmp_task_t*);
```

The task entry has signature `kmp_int32 entry(kmp_int32 gtid, void *task_t)`,
and `kmp_task_t` begins with a `void *shareds` pointer to the captured block.
The entry must load `shareds` from `task_t` before unpacking captures. This does
**not** fit the `void(void*)` closure path and needs a distinct outline shape.

### 2.3 pmsis — embedded cluster

PMSIS provides the fork/barrier team model (`ext_pi_cl_team_fork`,
`ext_pi_cl_team_barrier` — shims over the PMSIS API, see
[`pmsis-interface-adapter.c`](pmsis-interface-adapter.c)) but **no standard
deferred-task API**. A task mapping
must be defined explicitly (e.g. a custom `ext_pi_cl_task_push` / wait pair, or
declaring tasks unsupported on this target). No invented API is assumed here.

---

## 3. Pipeline overview

`omp.task` flows through the same three passes as the other constructs:

1. **OmpToOmpLowerPass** — walk `omp.task`, build the DSL `LoweringPlan`, and
   emit an `omp_lower.construct` carrying the plan, the body region, and the
   `if`-clause SSA value as an operand.
2. **OmpOutliningPass** — `outlineConstruct` outlines the body and emits the
   runtime call. For libgomp this reuses the packed/closure path.
3. **PlanLoweringPass** — unchanged (region-bearing constructs are fully lowered
   by the outlining pass).

---

## 4. DSL surface

### 4.1 libgomp (implemented)

```
construct task {
  outline_signature = closure(env_ptr);
  capture_strategy  = "packed";
  invoke {
    when has(if_clause) =>
      call "GOMP_task"(body, env_ptr, null,
                       env_size, env_align,
                       if_clause, 0, null, 0, null);
    otherwise =>
      call "GOMP_task"(body, env_ptr, null,
                       env_size, env_align,
                       true, 0, null, 0, null);
  }
}
```

New symbolic call-argument tokens (in addition to the existing `body`,
`env_ptr`, `num_threads`):

| DSL token   | resolves to (at call site)                          |
|-------------|-----------------------------------------------------|
| `env_size`  | `i64` constant = `sizeof(capture struct)`           |
| `env_align` | `i64` constant = `alignof(capture struct)` (≥ fallback 16) |
| `if_clause` | the `omp.task` `if` SSA value, widened to `i8`       |
| `null`      | a null pointer (`llvm.zero : !llvm.ptr`)            |
| `true`      | `i8` constant `1` (C `_Bool`)                        |

The `when has(if_clause)` / `otherwise` split is driven entirely by the
evaluation context (see §5.1); both branches share the same call shape except
for the boolean argument.

### 4.2 iomp (implemented — Approach A)

```
construct task {
  outline_signature = task_entry();               // ABI tag (i32(i32 gtid, ptr task)), matched head-only
  // no capture_strategy: the only valid topology (packed, runtime-allocated,
  // reached via task->shareds) is entailed by task_entry, so it would be inert.
  // no pre block: every invoke call uses both ident and global_tid (unlike
  // parallel, where gtid is optional), so there is no optionality to gate.
  // They are resolved on demand from the tokens below, as barrier/wsloop do.
  invoke {
    call "__kmpc_omp_task_alloc"(ident, global_tid, task_flags,
                                 task_size, shareds_size, body);
    call "__kmpc_omp_task"(ident, global_tid, task);
  }
}
```

New symbolic tokens (resolved in the outlining pass):

| Token | Resolves to |
|-------|-------------|
| `task_entry` | new outline signature `i32(i32 gtid, ptr task)` — **the dispatch discriminator** (no `capture_strategy`: the packed, runtime-allocated topology reached via `task->shareds` is entailed by the ABI) |
| `body` | the entry fn pointer (`kmp_routine_entry_t`) |
| `task_flags` | i32 flags — **v1 = 1 (tied)**; `final`/`untied`/… later |
| `task_size` | i64 `sizeof(kmp_task_t)` — `{ptr,ptr,i32,ptr,ptr}` = 40B header |
| `shareds_size` | i64 `sizeof(capture struct)` |
| `task` | the **result** of `__kmpc_omp_task_alloc` (Approach A, §5.3) |

**Result binding (Approach A).** `task` is *not* a context variable: the
outlining pass remembers the `__kmpc_omp_task_alloc` result, auto-inserts the
shareds-population block after it, and resolves `task` to that result. The
explicit `let task = call …` alternative (Approach B) is captured separately in
[dsl-result-binding-proposal.md](dsl-result-binding-proposal.md) for when a
second construct needs result threading.

**v1 simplifications:** all captures go into shareds (no separate clang-style
`privates` region); `flags = 1` (tied); the **`if` clause is deferred** — iomp
`if(cond)` needs a *runtime* branch to `__kmpc_omp_task_begin_if0` /
`__kmpc_omp_task_complete_if0`, unlike libgomp's plain boolean arg.

### 4.3 pmsis (planned)

Deferred until the embedded task API is chosen. If a closure-style push exists,
the shape mirrors libgomp (`closure` + `packed`) with a different callee.

---

## 5. Pass changes

### 5.1 `OmpToOmpLowerPass.cpp`

- **Collect** top-level `omp.task` (excluding tasks nested in a `parallel` or an
  enclosing `task`).
- **`extractTaskContext`** sets:
  - `body` → `"body"` (resolved to the outlined fn pointer downstream),
  - `env_ptr`, `packed` sentinels,
  - `env_size` / `env_align` placeholder sentinels,
  - `if_clause` → the sentinel string `"if_clause"` when `getIfExpr()` is set
    (non-null → `has(if_clause)` is true), else `null`.
  The actual `if` SSA value is **not** encoded in the context (its printed name
  is meaningless downstream); it rides as a real operand instead.
- **`ConstructOp` carries clause operands in a single `Variadic`
  (`clause_operands`)** rather than several `Optional` operands. At most one is
  present today — `num_threads` (parallel) or `if_clause` (task) — and the
  construct kind plus the plan's symbolic arg name disambiguate which it is.
  (Two `Optional` operands would need `AttrSizedOperandSegments`, whose
  generated verifier is broken when the dialect disables
  `usePropertiesForAttributes`, so a single variadic is used instead.)
  `emitConstructOp` / the `process` lambda thread the value through.
- **Shared `injectFirstprivateUses` helper** (used by both `parallel` and
  `task`) injects `unrealized_conversion_cast` uses of firstprivate source vars
  into the region so `collectCaptures` finds them.
- **Nested tasks (v2)** — *every* `omp.task` is collected, including those
  inside a `parallel` or another `task`. Parallels are converted first: moving a
  parallel body into a `ConstructOp` region carries any nested `omp.task` with
  it (the op pointer stays valid), and the task is then converted into a
  *nested* `ConstructOp`. Pre-order walk guarantees an outer task is processed
  before a task nested inside it.

### 5.2 `OmpOutliningPass.cpp`

`OmpOutliningPass` already `walk`s nested `ConstructOp`s and processes them in
pre-order, so the outer parallel is outlined before the inner task: the parallel
body (with the nested task `ConstructOp`) moves into `outlined_parallel_N`, the
parallel's `collectCaptures` pulls in anything the task uses from outside, and
`replaceUsesInRegion` rewrites those uses *inside* the task region too. When the
task is then outlined, its captures resolve to the unpacked values living in
`outlined_parallel_N`. No special nesting code is required beyond collecting the
tasks. (The shared `counter` is global, so the inner task's function is e.g.
`outlined_task_1` when the parallel took `0`.)


- Outlined function name is now `outlined_<construct>_N`
  (`outlined_parallel_N` unchanged; tasks become `outlined_task_N`).
- The capture-struct type (`structTy`) is hoisted so size/alignment can be
  computed once: via `mlir::DataLayout(module)` —
  `getTypeSize` / `getTypeABIAlignment`, with `alignof` falling back to `16`
  (a valid power-of-2 ≥ any field's alignment; safe because `cpyfn == NULL`).
- The packed-invoke argument resolver gains cases for `env_size`, `env_align`,
  `if_clause` (i1→i8 via `zext`/`trunc`), `null` (→ `llvm.zero`), and a
  `BoolAttr` branch (→ `i8` constant). `BoolAttr` is matched **before**
  `IntegerAttr` because MLIR's `BoolAttr` is an `i1` `IntegerAttr`.
- `GOMP_task` is **not** variadic → the ordinary `getOrInsertDecl` path applies
  (the `__kmpc_fork_call` variadic special-case is skipped). Declaration arg
  types are inferred from the SSA values, yielding the §2.1 ABI.

### 5.3 iomp (implemented — Approach A)

A new outlining branch (`outlineTaskShareds`) keyed on `outline_signature`
containing `task_entry` (the capture topology stays `packed`). It cannot share
the `void(void*)` closure path. Concretely:

- **Entry function** `i32(i32 gtid, ptr task)`: load `shareds` from the task
  header (`GEP task[0,0]` → load ptr), then unpack each capture from the shareds
  struct (same struct as the libgomp packed strategy, but the base pointer is
  the loaded `shareds`, not the function arg). `omp.terminator` → `func.return`
  of `i32 0`.
- **Call site**, iterating *all* invoke calls (not just the first):
  - `task_size` → `sizeof(kmp_task_t)` where the header struct is
    `{ptr,ptr,i32,ptr,ptr}` (40 B) to match the runtime; `shareds_size` →
    `sizeof(capture struct)`; `task_flags` → `i32 1`; `body` → entry fn ptr.
  - Emit `__kmpc_omp_task_alloc(...)` and **bind its result** to the `task`
    token (Approach A).
  - **Populate shareds**: `GEP task[0,0]` → load `shareds`; for each capture
    `GEP shareds[0,i]` + store. (C++-generated; inexpressible in the DSL.)
  - Emit `__kmpc_omp_task(ident, gtid, task)`.
- `ident` / `global_tid` are resolved on demand at the call site (default ident +
  a memoised `__kmpc_global_thread_num`), the same resolve-on-reference model
  barrier/wsloop use — not the parallel `emit`-gated pre block, since a task
  always needs both.

See [dsl-result-binding-proposal.md](dsl-result-binding-proposal.md) for the
Approach B (`let = call`) alternative to the implicit `task` binding.

### 5.4 pmsis (planned)

Deferred until the embedded task API is chosen.

---

## 6. Worked example (libgomp)

Input:
```mlir
func.func @task_if(%arg0: !llvm.ptr, %cond: i1) {
  omp.task if(%cond) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}
```

After `--omp-to-omp-lower --omp-outline` (sketch):
```mlir
// Body outlined into a closure; %arg0 unpacked from the capture struct.
func.func nested @outlined_task_0(%data: !llvm.ptr) {
  %p = llvm.load %data ... : !llvm.ptr        // unpack capture
  llvm.call @use(%p) : (!llvm.ptr) -> ()
  return
}

func.func @task_if(%arg0: !llvm.ptr, %cond: i1) {
  // build { ptr } capture struct on the stack, store %arg0
  %if8 = llvm.zext %cond : i1 to i8
  %sz  = llvm.mlir.constant(8 : i64) : i64    // sizeof struct
  %al  = llvm.mlir.constant(8 : i64) : i64    // alignof struct
  %fn  = ... @outlined_task_0 as ptr
  %nl  = llvm.zero : !llvm.ptr
  %z32 = ... 0 : i32
  call @GOMP_task(%fn, %env, %nl, %sz, %al, %if8, %z32, %nl, %z32, %nl)
  return
}
```

The `otherwise` branch (no `if`) is identical except the boolean argument is the
`i8` constant `1`.

---

## 7. Testing

- **`test/Regression/task-libgomp.mlir`** — `--omp-to-omp-lower` on an empty
  task: checks `omp_lower.construct` with `runtime = "libgomp"`,
  `construct = "task"`, and `GOMP_task` in the invoke plan.
- **`test/Regression/task-outline-libgomp.mlir`** — full
  `--omp-to-omp-lower --omp-outline` on a task with a capture and an `if`
  clause: checks the `outlined_task_0` closure, the `i1 → i8` widening of the
  if-clause, and the `GOMP_task` call.
- **`test/Regression/task-nested-libgomp.mlir`** — a `parallel { task }`:
  checks the task is outlined into its own closure, the `GOMP_task` call lands
  inside the parallel's outlined function, and the outer function forks via
  `GOMP_parallel`.
- **`test/Integration/tasks/run_tasks.sh`** — two end-to-end checks against
  real libgomp, both expecting `42`:
  - `tasks/task_nested.mlir` — hand-written `parallel { task { *p = 42 } }`
    lowered + linked + run. MLIR input, so independent of the CIR front-end.
  - `tasks/task_smoke.c` — same program in C, built with `gcc -fopenmp` (ref)
    and through the full CIR / `mlir-opt-omp` pipeline (opt); outputs must
    match. Depends on ClangIR emitting `omp.task`.
- **`test/Regression/task-iomp.mlir`** — an iomp task: checks the
  `i32(i32 gtid, ptr task) -> i32` entry and the `__kmpc_global_thread_num` /
  `__kmpc_omp_task_alloc` / `__kmpc_omp_task` call sequence.
- Future (iomp): the `if0` begin/complete (`if`/`final`) path; firstprivate via
  shareds; an end-to-end run against `libomp`.

---

## 8. Non-goals / deferred

- **`depend`, `priority`, `detach`, `untied`, `mergeable`, `final`** clauses —
  currently hard-wired (`depend = null`, `priority = 0`, `detach = null`,
  `flags = 0`). Additive later.
- **`taskwait` / `taskgroup` / `taskloop`** — separate constructs, out of scope.
- **LLP64 targets** — the `long = i64`, `_Bool = i8` mapping assumes LP64
  (the wsl/workstation Linux targets). Revisit only for a Windows/LLP64 libgomp.
- **iomp `_Bool`/`zeroext` ABI nuances** and **pmsis task API** — see §2.2/§2.3.

---

## 9. Implementation order

1. **libgomp v1** — DSL `construct task`; `ConstructOp` clause operand;
   `extractTaskContext` + top-level task collection; packed-invoke resolver
   cases (`env_size`/`env_align`/`if_clause`/`null`/bool); DataLayout size/align;
   `null` DSL literal; regression tests. **(done)**
2. **libgomp v2** — nested tasks (collect every `omp.task`); nested-task
   regression test; end-to-end `run_tasks.sh` integration test. **(done)**
3. **iomp** — new `task_entry`/`shareds` outline branch
   (`outlineTaskShareds`); `__kmpc_omp_task_alloc` (result bound to `task`,
   Approach A) + shareds population + `__kmpc_omp_task`; `task-iomp.mlir`.
   **(implemented — pending build/test)**. Follow-ups: `if`/`final` (`if0`
   path), firstprivate via shareds, libomp end-to-end run.
4. **pmsis** — define the embedded task API, then map (closure-style if
   available); tests.
