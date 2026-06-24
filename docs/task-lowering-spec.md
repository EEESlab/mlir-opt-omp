# Specification: `omp.task` lowering across runtimes

Scope: lowering of the OpenMP **task** construct (`omp.task`) to the three
supported runtimes — **libgomp** (`GOMP_task`), **iomp** (`__kmpc_omp_task*`),
and **pmsis** (embedded cluster API).

Status:

| Runtime | Status        |
|---------|---------------|
| libgomp | **implemented** (v1) |
| iomp    | planned (phase 2) |
| pmsis   | planned (phase 3, API to be defined) |

Decisions baked into v1: **closure/packed reuse** for libgomp, **DataLayout
with fallback-16** for the env-struct alignment, and **top-level tasks only**
(tasks nested in a `parallel` or another `task` are a documented follow-up).

---

## 1. Goal

Lower `omp.task` so that the captured environment is snapshotted and the task
body runs through the selected runtime's task API, matching each runtime's ABI:

1. The task body is **outlined** into a function whose signature matches the
   runtime's task-entry convention.
2. The captured variables are packaged per the runtime's calling convention
   (a copied closure struct for libgomp, a `kmp_task_t` shareds block for iomp).
3. The `if` clause selects the deferred-vs-undeferred path where the runtime
   distinguishes them.

OpenMP semantics to honour: variables referenced in a task are **firstprivate
by default**, i.e. captured *by value* at task-creation time. The existing
capture-packing machinery (scalar/pointer allocas packed by value) already
provides these snapshot semantics; explicit `private`/`firstprivate` recipes
are handled via the shared firstprivate-injection step.

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
                int   priority);
```

Key behaviour: with `cpyfn == NULL`, libgomp allocates `arg_size` bytes aligned
to `arg_align`, `memcpy`s `data` into that task-private block, and later calls
`fn(copy)`. This is true even for undeferred tasks (`if_clause == false` /
`final`). Therefore the outlined body is an ordinary closure
`void (*)(void *)` — **identical to the libgomp `parallel` closure** — and the
only task-specific data is `arg_size` / `arg_align`.

ABI mapping used by the tool (Linux x86-64 / LP64):

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
`ext_pi_cl_team_barrier`) but **no standard deferred-task API**. A task mapping
must be defined explicitly (e.g. a custom `ext_pi_cl_task_push` / wait pair, or
declaring tasks unsupported on this target). No invented API is assumed here.

---

## 3. Pipeline overview

`omp.task` flows through the same three passes as the other constructs:

1. **OmpToOmpLowerPass** — walk `omp.task`, build the DSL `LoweringPlan`, and
   emit an `omp_lower.construct` carrying the plan, the body region, and the
   `if`-clause SSA value as an operand.
2. **OmpOutliningPass** — `outlineConstruct` outlines the body and emits the
   runtime call. For libgomp this reuses the packed/closure path verbatim.
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
                       if_clause, 0, null, 0);
    otherwise =>
      call "GOMP_task"(body, env_ptr, null,
                       env_size, env_align,
                       true, 0, null, 0);
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

### 4.2 iomp (planned)

```
construct task {
  outline_signature = task_entry(gtid, task_t);   // kmp_int32(kmp_int32, void*)
  capture_strategy  = "shareds";                  // captures live in task_t->shareds
  pre {
    emit ident;
    emit global_tid;
    // task = __kmpc_omp_task_alloc(ident, gtid, flags,
    //                              sizeof(kmp_task_t), env_size, body);
    // copy captures into task->shareds
  }
  invoke {
    when has(if_clause) =>   // if(0)/final → run inline
      // begin_if0; call task->routine; complete_if0
      ...;
    otherwise =>
      call "__kmpc_omp_task"(ident, global_tid, task);
  }
}
```

This introduces a new `outline_signature` (`task_entry`) and a new
`capture_strategy` (`shareds`) that the outlining pass must learn.

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

### 5.2 `OmpOutliningPass.cpp`

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

### 5.3 iomp/pmsis (planned)

iomp needs a new outlining branch for `task_entry(gtid, task_t)` /
`capture_strategy = "shareds"`: allocate via `__kmpc_omp_task_alloc`, copy the
captures into `task->shareds`, then `__kmpc_omp_task` (or the `if0`
begin/complete pair). This cannot share the `void(void*)` closure path.

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
  call @GOMP_task(%fn, %env, %nl, %sz, %al, %if8, %z32, %nl, %z32)
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
- Future (iomp): `__kmpc_omp_task_alloc` + `__kmpc_omp_task` emission, the
  `task_entry(gtid, task_t)` signature, and the `if0` begin/complete path.

---

## 8. Non-goals / deferred

- **Nested tasks** (inside a `parallel` or another `task`). v1 handles only
  top-level tasks; nested support depends on outlining order of nested
  construct ops and needs dedicated tests.
- **`depend`, `priority`, `untied`, `mergeable`, `final`** clauses — currently
  hard-wired (`depend = null`, `priority = 0`, `flags = 0`). Additive later.
- **`taskwait` / `taskgroup` / `taskloop`** — separate constructs, out of scope.
- **LLP64 targets** — the `long = i64`, `_Bool = i8` mapping assumes LP64
  (the wsl/workstation Linux targets). Revisit only for a Windows/LLP64 libgomp.
- **iomp `_Bool`/`zeroext` ABI nuances** and **pmsis task API** — see §2.2/§2.3.

---

## 9. Implementation order

1. **libgomp** — DSL `construct task`; `ConstructOp` `if_clause` operand
   (+`AttrSizedOperandSegments`); `extractTaskContext` + task collection;
   packed-invoke resolver cases (`env_size`/`env_align`/`if_clause`/`null`/bool);
   DataLayout size/align; regression tests. **(done)**
2. **iomp** — new `task_entry`/`shareds` outline branch; `task_alloc` +
   `task`/`if0` lowering; flag tokens; tests.
3. **pmsis** — define the embedded task API, then map (closure-style if
   available); tests.
