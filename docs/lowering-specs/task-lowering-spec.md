# Specification: `omp.task` lowering across runtimes

Status:

| Runtime | Status        |
|---------|---------------|
| libgomp | **implemented** (v2) |
| iomp    | **implemented** (tied tasks + explicit firstprivate + `if`; `final` deferred) |
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
`ext_pi_cl_team_barrier` — shims over the PMSIS API, provided and linked by the
PULP harness) but **no standard deferred-task API**. A task mapping
must be defined explicitly (e.g. a custom `ext_pi_cl_task_push` / wait pair, or
declaring tasks unsupported on this target). No invented API is assumed here.

---

## 3. Pipeline overview

`omp.task` flows through the same three passes as the other constructs:

1. **OmpToOmpLowerPass** — walk `omp.task`, build the DSL `LoweringPlan`, and
   emit an `omp_lower.construct` carrying the plan, the body region, and the
   `if`-clause SSA value as an operand.
2. **OmpOutliningPass** — outlines the body and attaches to the construct what
   only it can produce: the entry pointer, the capture struct or the resolved
   capture values, the ABI sizes. For libgomp this reuses the packed/closure
   path. It emits no runtime call and leaves the construct standing.
3. **PlanLoweringPass** — emits the whole call sequence from the plan, including
   the `if` branch and, for iomp, the write into `task->shareds`.

---

## 4. DSL surface

### 4.1 libgomp (implemented)

```
construct task {
  capture_strategy = packed;   // closure signature void(ptr data), one struct
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

### 4.2 iomp (implemented — Approach B)

```
construct task {
  capture_strategy = shareds;   // ABI selector: task-routine i32(i32 gtid, ptr task),
                                // captures runtime-allocated, reached via task->shareds
  kmp_task_t = struct(ptr, ptr, i32, ptr, ptr);
  let task_flags = 1;
  // no pre block: every invoke call uses both ident and global_tid (unlike
  // parallel, where gtid is optional), so there is no optionality to gate.
  // They are resolved on demand from the tokens below, as barrier/wsloop do.
  invoke {
    let task = call "__kmpc_omp_task_alloc"(ident, global_tid, task_flags,
                                            task_size, shareds_size, body);
    emit populate_shareds(task);
    branch if_clause {
      true  => call "__kmpc_omp_task"(ident, global_tid, task);
      false => {
        call "__kmpc_omp_task_begin_if0"(ident, global_tid, task);
        call body(global_tid, task);
        call "__kmpc_omp_task_complete_if0"(ident, global_tid, task);
      }
    }
  }
}
```

Symbolic tokens, all bound by the outlining pass and resolved by the plan pass:

| Token | Resolves to |
|-------|-------------|
| `capture_strategy = shareds` | selects the task-routine signature `i32(i32 gtid, ptr task)` — **the dispatch discriminator**; the packed, runtime-allocated topology reached via `task->shareds` is entailed by this ABI value |
| `body` | the entry fn pointer (`kmp_routine_entry_t`); also usable as a callee, for the direct call on the undeferred side |
| `task_flags` | i32 flags — **v1 = 1 (tied)**; `final`/`untied`/… later |
| `task_size` | i64 `sizeof(kmp_task_t)` — `{ptr,ptr,i32,ptr,ptr}` = 40B header |
| `shareds_size` | i64 `sizeof(capture struct)` |
| `task` | the **result** of `__kmpc_omp_task_alloc`, bound by the `let` |

**Result binding.** `let <name> = call …` binds the call's SSA result under
`%<name>`; every later argument naming it resolves to that value. It was chosen
over an implicit `task` token so that nothing in the DSL is named by convention.

**`if` clause.** iomp has no boolean parameter for it, so the clause becomes a
`branch` on the runtime value: true schedules the task deferred, false runs it
on the spawning thread between `__kmpc_omp_task_begin_if0` and
`__kmpc_omp_task_complete_if0`. The task is allocated and populated either way —
the undeferred calls take a `kmp_task_t*` too. With no `if` clause the condition
is null, the branch collapses at evaluation time, and only the deferred call is
emitted.

**v1 simplifications:** all captures go into shareds (no separate clang-style
`privates` region); **explicit firstprivate** is copied into a task-private slot
in the entry prolog (see §5.3), but a pure `private` clause is not wired (it is
diagnosed, not miscompiled); `flags = 1` (tied).

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
  (Two `Optional` operands would need `AttrSizedOperandSegments` and the
  segment-size attribute it brings along, so a single variadic is used
  instead.)
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

### 5.3 iomp (implemented — Approach B)

A separate outlining branch (`outlineTaskEntry`) keyed on
`capture_strategy == shareds` (the capture topology is a runtime-allocated
struct). It cannot share the `void(void*)` closure path. Concretely:

- **Entry function** `i32(i32 gtid, ptr task)`: load `shareds` from the task
  header (`GEP task[0,0]` → load ptr), then unpack each capture from the shareds
  struct (same struct as the libgomp packed strategy, but the base pointer is
  the loaded `shareds`, not the function arg). `omp.terminator` → `func.return`
  of `i32 0`.
- **Explicit firstprivate**: the privatizer source ptrs are injected as leading
  captures by `OmpToOmpLowerPass` (shared with parallel), so after unpacking,
  `loadedCaptures[i]` is the source slot for the *i*-th privatizer block arg. For
  each, the prolog allocates a task-private slot, copies the source into it, and
  rewrites the block arg's uses to it. The leftover (now-dead) block args are
  dropped so the entry keeps its two-parameter ABI; a block arg that survives
  with live uses (a pure `private` clause, or a firstprivate whose element type
  can't be inferred) is diagnosed rather than emitted as a wrong-ABI entry.
- **Snapshot timing**: a firstprivate value must be captured *by value* so it is
  snapshotted into shareds at task **creation**, not read through a captured
  pointer at task **entry** — otherwise a deferred task that runs after the
  source was mutated (the canonical `firstprivate(i)` spawn loop) observes the
  wrong value. `classifyCaptures` leaves the source as a plain (pointer) capture
  because its first in-region use is the injected marker cast, so
  the shared `forceFirstprivateByValue` helper re-classifies scalar-**alloca**
  sources into the by-value bucket, making `resolveCaptureValues` load and
  snapshot them at the call site. It runs on both the iomp (`outlineTaskEntry`)
  and the libgomp/packed (`outlineConstruct`) task paths; for `parallel` it is
  harmless because creation coincides with the fork. Non-alloca (e.g. by-pointer
  argument) sources cannot be snapshotted this way and keep the read-at-entry
  behaviour — a documented limitation.
- **Call site.** The outlining pass emits no call. It binds to the construct
  what only it knows and leaves it standing:
  - `task_size` → `sizeof(kmp_task_t)`, the header struct being
    `{ptr,ptr,i32,ptr,ptr}` (40 B) to match the runtime; `shareds_size` →
    `sizeof(capture struct)`; `body` → the entry fn ptr; `%captures` → the
    *resolved* per-field values (private → undef, scalar/ptr alloca → loaded by
    value, plain → as-is), because that classification is outlining knowledge.
    `task_flags` is not bound: it is a `let` in the DSL.
  - Everything else is `PlanLoweringPass`, from the plan: the
    `__kmpc_omp_task_alloc` call with its result bound to `task`
    (`let task = call ...`, Approach B); `emit populate_shareds(task)`, which
    reaches `GEP task[0,0]` → load `shareds` and stores the bound values field
    by field; and the `branch if_clause` carrying the deferred call on one side
    and the `begin_if0` / direct entry call / `complete_if0` protocol on the
    other. The `kmp_task_t` layout the GEP needs is the DSL property, read by
    both passes through the same helper.
- `ident` / `global_tid` are resolved on demand by the plan pass (default ident +
  a memoised `__kmpc_global_thread_num`), the same resolve-on-reference model
  parallel/barrier/wsloop use — no `emit` declaration anywhere.

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

- **`test/Regression/task/task-libgomp.mlir`** — `--omp-to-omp-lower` on an empty
  task: checks `omp_lower.construct` with `runtime = "libgomp"`,
  `construct = "task"`, and `GOMP_task` in the invoke plan.
- **`test/Regression/task/task-outline-libgomp.mlir`** — full
  `--omp-to-omp-lower --omp-outline` on a task with a capture and an `if`
  clause: checks the `outlined_task_0` closure, the `i1 → i8` widening of the
  if-clause, and the `GOMP_task` call.
- **`test/Regression/task/task-nested-libgomp.mlir`** — a `parallel { task }`:
  checks the task is outlined into its own closure, the `GOMP_task` call lands
  inside the parallel's outlined function, and the outer function forks via
  `GOMP_parallel`.
- **`test/Integration/tasks/run_tasks.sh`** — six end-to-end checks against
  the real runtime (`./run_tasks.sh [libgomp|iomp]`), all expecting `42`:
  - `tasks/task_nested.mlir` — hand-written `parallel { task { *p = 42 } }`
    lowered + linked + run. MLIR input, so independent of the CIR front-end.
  - `tasks/task_smoke.c` — same program in C, built with `gcc -fopenmp` (ref)
    and through the full CIR / `mlir-opt-omp` pipeline (opt); outputs must
    match. Depends on ClangIR emitting `omp.task`.
  - `tasks/taskwait_nested.mlir` — hand-written `parallel { task; taskwait;
    read-back }` where the taskwait is *load-bearing* (the task's write is read
    back after the taskwait, inside the region), exercising `GOMP_taskwait` /
    `__kmpc_omp_taskwait` end to end. MLIR input, front-end independent.
  - `tasks/taskwait_smoke.c` — the same load-bearing taskwait in C, ref vs opt
    (like `task_smoke.c`). Depends on ClangIR emitting `omp.taskwait`.
  - `tasks/task_firstprivate.mlir` — a `parallel { task }` with an *explicit*
    firstprivate clause; prints `42` iff the firstprivate copy-in ran. Passes on
    both libgomp (packed path) and iomp (`outlineTaskEntry` copy-in). Nested in
    a parallel so the implicit barrier completes the task (keeps it focused on
    firstprivate; top-level taskwait is covered separately).
  - `tasks/taskwait_toplevel.mlir` — a top-level task + top-level `omp.taskwait`
    (outside any parallel); prints `42`. Exercises the plan pass materialising a
    real gtid from `__kmpc_global_thread_num`, the path that previously crashed
    iomp.
- **`test/Regression/task/task-iomp.mlir`** — an iomp task: checks the
  `i32(i32 gtid, ptr task) -> i32` entry and the `__kmpc_global_thread_num` /
  `__kmpc_omp_task_alloc` / `__kmpc_omp_task` call sequence.
- **`test/Regression/taskwait/taskwait-toplevel-iomp.mlir`** — a top-level `omp.taskwait`
  lowers (in the outlining pass) to `__kmpc_global_thread_num` +
  `__kmpc_omp_taskwait` with a real gtid, leaving no `omp_lower.construct`.
- **`test/Regression/task/firstprivate/task-firstprivate-{iomp,libgomp}.mlir`** — a task with an
  explicit firstprivate clause: checks the copy-in (a task-private `llvm.alloca`)
  and that no firstprivate parameter leaks into the outlined signature.
- **`test/Regression/task/firstprivate/task-firstprivate-snapshot-{iomp,libgomp}.mlir`** — the
  snapshot-timing property: the scalar firstprivate source is loaded by value at
  the call site (task creation), not dereferenced at entry. Both runtimes pass.
- **`test/Regression/task/firstprivate/task-firstprivate-unsupported-{iomp,libgomp}.mlir`** — an
  unsupported firstprivate shape (block arg used without a scalar load, so the
  element type can't be inferred) is diagnosed on both paths (`-verify-diagnostics`)
  instead of silently emitting a wrong-ABI outlined function.
- **`test/Regression/{task,parallel}/if/`** — the `if` clause
  on both constructs and both runtimes. iomp branches on the condition at the
  call site: a task takes the undeferred `__kmpc_omp_task_begin_if0` / direct
  entry call / `_complete_if0` protocol, a parallel takes
  `__kmpc_serialized_parallel` + a direct microtask call. libgomp needs no
  branch: the task hands the condition to `GOMP_task` as its `_Bool` parameter
  (i1 zero-extended to i8), and the parallel forces `num_threads` to 1.
- Future (iomp): `final`, which reuses the same if0 machinery; pure `private`
  clause wiring.

---

## 8. Non-goals / deferred

- **`depend`, `priority`, `detach`, `untied`, `mergeable`, `final`** clauses —
  currently hard-wired (`depend = null`, `priority = 0`, `detach = null`,
  `flags = 0`). Additive later.
- **`taskwait`** — implemented for iomp and libgomp as a leaf construct,
  mirroring `barrier`: no body, no captures. iomp emits
  `__kmpc_omp_taskwait(ident, gtid)` with — like clang's `emitTaskwaitCall` — a
  plain default `ident` (flags = `KMPC` = 0x02; there is no taskwait-specific
  `OpenMPLocationFlags` bit); libgomp emits the no-argument `GOMP_taskwait()`.
  The pass logic is runtime-agnostic (plan built from the DSL), so the
  `taskwait` plan is optional: a `taskwait` under a runtime lacking the
  construct is diagnosed, not dropped. A `taskwait` nested directly in a task
  body (`parallel { task { taskwait } }`) is lowered by `PlanLoweringPass` like
  any other leaf, but the outlining pass has to bind its `%gtid` first: there
  the thread id is the entry's arg 0 *by value* (the shareds ABI
  `i32(i32 gtid, ptr task)`), not the microtask ptr-to-i32. An `omp.barrier`
  inside a task region is invalid OpenMP and is diagnosed there.
  A *top-level* `taskwait`/`barrier` (not inside a parallel) has no such source,
  so the plan pass materialises a real gtid from `__kmpc_global_thread_num`
  itself. In-parallel taskwait/barrier take the real `%gtid` from the microtask
  arg.
  `depend`/`nowait` on taskwait are still ignored (a warning is emitted);
  pmsis (no standard task-wait API) is a follow-up.
- **`taskgroup` / `taskloop`** — separate constructs, out of scope.
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
   (`outlineTaskEntry`); `__kmpc_omp_task_alloc` (result bound to `task`,
   Approach B) + shareds population + `__kmpc_omp_task`; `task-iomp.mlir`.
   **(done)**. Explicit firstprivate copy-in + `task-firstprivate-*.mlir` and
   the `task_firstprivate.mlir` integration case, and scalar firstprivate
   snapshot-at-creation timing (both runtimes, via the shared
   `forceFirstprivateByValue` helper). **(done)**. `if` via the `if0` path,
   stated as a `branch` in the DSL and emitted by `PlanLoweringPass` together
   with the rest of the call sequence. **(done)**. Follow-ups: `final`, pure
   `private` clause wiring, by-pointer firstprivate snapshot.
4. **pmsis** — define the embedded task API, then map (closure-style if
   available); tests.
