# Proposal: `let = call` result binding in the lowering DSL

The iomp `task` lowering uses implicit result binding in C++ — see [task-lowering-spec.md](task-lowering-spec.md). 

This document records another possible approach so it can be picked up when a
*second* construct needs to thread a call's result, without re-deriving the
design.

---

## 1. Problem

Some runtime sequences feed the **result of one call into a later call**. The
motivating case is iomp tasks:

```c
kmp_task_t *t = __kmpc_omp_task_alloc(loc, gtid, flags, tsz, ssz, &entry);
/* populate t->shareds */
__kmpc_omp_task(loc, gtid, t);          // uses t
```

Today the DSL's `call` is **fire-and-forget**: it lowers to a `func.call` whose
result is dropped. There is no way in the DSL to name a call's result and refer
to it from a later argument.

For iomp `task` we worked around this: the C++ lowering for
`capture_strategy = "shareds"` recognises the `__kmpc_omp_task_alloc` callee,
remembers its SSA result, and resolves a magic `task` token to it. Zero
parser changes but the data flow is invisible in the DSL and the binding is
hardcoded to one callee.

An **alternative approach** `let = call` makes that data flow explicit and reusable.

---

## 2. Proposed DSL

```
invoke {
  let task = call "__kmpc_omp_task_alloc"(ident, global_tid, task_flags,
                                          task_size, shareds_size, body);
  emit populate_shareds(task);          // still a C++-backed verb (see §5)
  call "__kmpc_omp_task"(ident, global_tid, task);
}
```

`let <name> = call "<callee>"(<args>);` binds `<name>` to the call's SSA result;
later statements reference `<name>` like any other symbol.

---

## 3. Parser / AST changes

Today (`DSLParser.h`):
- `LetDecl { name; Expr expr; }` — `let` binds an **expression**.
- `Action = EmitAction | CallAction` — a `call` is an **action**, not an
  expression, so `let x = call ...` does not parse.

Two ways to allow it:

- **(a) `call` as an expression.** Add a `CallRuntimeExpr` variant. Risk:
  ambiguity with the existing builtin `CallExpr` (e.g. `closure(...)`,
  `argc(...)`), which are evaluated, not emitted. Would need the lexer/parser to
  distinguish `call "str"(...)` (string callee ⇒ runtime call) from
  `ident(...)` (builtin).
- **(b) Dedicated `LetCallStmt` (recommended).** A new statement form parsed
  only inside blocks: `let NAME = call STRING ( arglist )`. Keeps the builtin
  `CallExpr` grammar untouched; the `call` keyword + string callee is an
  unambiguous trigger.

Recommended: **(b)**. New `Statement` alternative:

```cpp
struct LetCallStmt { std::string name; CallAction call; };
```

---

## 4. Evaluator / plan changes

- `PlanCall` gains an optional `std::string resultName;`.
- When evaluating a `LetCallStmt`: emit the `PlanCall` with `resultName = name`,
  and bind `name` in the current scope to a symbolic placeholder (e.g.
  `StrVal("%" + name)`) so later argument expressions referencing it resolve to
  that token.
- `PlanCallAttr` (TableGen) gains an optional result-symbol parameter so the
  binding survives into `omp_lower.construct`. `dslValueToAttr` /
  `planActionToAttr` carry it through.

---

## 5. C++ outlining changes

- Maintain a `llvm::StringMap<Value>` of bound results within a construct's
  lowering.
- When emitting a `PlanCall` that has a `resultName`, store its `func.call`
  result under that name.
- Token resolution (the `resolveArg`-style helpers) consults that map first, so
  `task` resolves to the `task_alloc` result.

**Escape hatch still required.** Buffer/struct population — e.g. writing the
captures into `task->shareds` (GEP + load + store per field) — **cannot** be
expressed in the DSL. It remains a C++-backed marker, surfaced as
`emit populate_shareds(<task>)` (an `EmitAction` the lowering recognises). So
`let = call` makes the *control/data flow* explicit but does **not** remove the
need for a few opaque, runtime-specific verbs.

---

## 6. When to adopt

Adopt when a **second** construct needs result threading (candidates: iomp
`detach`/`taskgroup`, or another runtime returning a handle). At that point,
refactor the iomp `task` Approach-A binding into this explicit form — the
**target IR does not change**, only the DSL surface and where the binding is
expressed. Until then, Approach A keeps the surface minimal for the single
current user.

---

## 7. Cost summary

| Change | Effort |
|--------|--------|
| Parser: `LetCallStmt` | small |
| Evaluator: `resultName` + scope binding | small |
| `PlanCallAttr`: optional result symbol (TableGen + converters) | small |
| Outlining: name→SSA map + resolution | small |
| `populate_shareds` (needed by Approach A too) | shared, not extra |

Total: medium, mostly mechanical — but only worthwhile once it has more than one
caller.
