// DSLEvaluator.cpp
//
// C++ port of evaluator.py.

#include "DSLEvaluator.h"
#include "DSLParser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <sstream>

using namespace dsl;
using llvm::Expected;
using llvm::Error;
using llvm::StringRef;
using llvm::make_error;
using llvm::StringError;
using llvm::inconvertibleErrorCode;

// ===========================================================================
// Value helpers
// ===========================================================================

bool dsl::isTruthy(const Value &v) {
  return std::visit(llvm::makeVisitor(
    [](const NullVal &)  { return false; },
    [](const BoolVal &b) { return b.value; },
    [](const IntVal &i)  { return i.value != 0; },
    [](const StrVal &s)  { return !s.value.empty(); },
    [](const ListVal &l) { return !l.items.empty(); }
  ), v);
}

std::string dsl::valueToString(const Value &v) {
  return std::visit(llvm::makeVisitor(
    [](const NullVal &)  -> std::string { return "null"; },
    [](const BoolVal &b) -> std::string { return b.value ? "true" : "false"; },
    [](const IntVal &i)  -> std::string { return std::to_string(i.value); },
    [](const StrVal &s)  -> std::string { return s.value; },
    [](const ListVal &l) -> std::string {
      std::string r = "[";
      auto items = listItems(l);
      for (size_t i = 0; i < items.size(); i++) {
        if (i) r += ", ";
        r += valueToString(items[i]);
      }
      r += "]";
      return r;
    }
  ), v);
}

// Equality comparison for Value
static bool valuesEqual(const Value &a, const Value &b) {
  if (a.index() != b.index()) return false;
  return std::visit(llvm::makeVisitor(
    [](const NullVal &, const NullVal &) { return true; },
    [](const BoolVal &x, const BoolVal &y) { return x.value == y.value; },
    [](const IntVal  &x, const IntVal  &y) { return x.value == y.value; },
    [](const StrVal  &x, const StrVal  &y) { return x.value == y.value; },
    [](const ListVal &, const ListVal &)   { return false; }, // not compared
    [](const auto &, const auto &) { return false; }
  ), a, b);
}

// ===========================================================================
// Scope implementation (child creation)
// ===========================================================================
// We implement Scope as a flat map with parent pointer.
// The makeChild helper creates a new Scope whose parent is the given scope.

class ScopeImpl {
public:
  ScopeImpl *parent;
  llvm::StringMap<Value> bindings;

  explicit ScopeImpl(ScopeImpl *p = nullptr) : parent(p) {}

  bool has(StringRef name) const {
    if (bindings.count(name)) return true;
    return parent && parent->has(name);
  }

  Expected<Value> get(StringRef name) const {
    auto it = bindings.find(name);
    if (it != bindings.end()) return it->second;
    if (parent) return parent->get(name);
    return make_error<StringError>(
      "unknown identifier: " + name.str(), inconvertibleErrorCode());
  }

  void set(StringRef name, Value v) { bindings[name] = std::move(v); }
};

// Helper to build a root scope from a context map
static ScopeImpl buildRootScope(const llvm::StringMap<Value> &ctx) {
  ScopeImpl s;
  for (auto &kv : ctx)
    s.set(kv.first(), kv.second);
  return s;
}

// ===========================================================================
// Evaluator internals
// ===========================================================================

// Forward declare the actual implementation using ScopeImpl
namespace {

Expected<Value> evalExpr(const Expr &expr, ScopeImpl &scope);
Expected<Value> evalExprOrBare(const Expr &expr, ScopeImpl &scope);
Expected<bool>  evalPredicate(const Predicate &pred, ScopeImpl &scope);
Expected<PlanAction> evalAction(const Action &action, ScopeImpl &scope);

// ---- Builtin calls --------------------------------------------------------

Expected<Value> evalBuiltin(const std::string &name, std::vector<Value> args) {
  if (name == "argc" || name == "len") {
    if (args.size() != 1)
      return make_error<StringError>(name + "() expects 1 arg", inconvertibleErrorCode());
    if (auto *l = std::get_if<ListVal>(&args[0]))
      return makeInt((int)l->items.size());
    return make_error<StringError>(name + "() expects a list", inconvertibleErrorCode());
  }
  if (name == "first") {
    if (args.size() != 1)
      return make_error<StringError>("first() expects 1 arg", inconvertibleErrorCode());
    if (auto *l = std::get_if<ListVal>(&args[0])) {
      if (l->items.empty())
        return make_error<StringError>("first() on empty list", inconvertibleErrorCode());
      return l->items[0]->v;
    }
    return make_error<StringError>("first() expects a list", inconvertibleErrorCode());
  }
  if (name == "concat") {
    ListVal out;
    for (auto &a : args) {
      if (auto *l = std::get_if<ListVal>(&a))
        for (auto &p : l->items) out.items.push_back(p);
      else
        out.items.push_back(std::make_shared<ValueBox>(ValueBox{a}));
    }
    return Value{out};
  }
  if (name == "microtask") {
    std::string r = "microtask(";
    for (size_t i = 0; i < args.size(); i++) {
      if (i) r += ", ";
      r += valueToString(args[i]);
    }
    r += ")";
    return makeStr(r);
  }
  if (name == "closure") {
    std::string r = "closure(";
    for (size_t i = 0; i < args.size(); i++) {
      if (i) r += ", ";
      r += valueToString(args[i]);
    }
    r += ")";
    return makeStr(r);
  }
  // Symbolic fallback
  std::string r = name + "(";
  for (size_t i = 0; i < args.size(); i++) {
    if (i) r += ", ";
    r += valueToString(args[i]);
  }
  r += ")";
  return makeStr(r);
}

// ---- Expr -----------------------------------------------------------------

Expected<Value> evalExpr(const Expr &expr, ScopeImpl &scope) {
  return std::visit(llvm::makeVisitor(
    [&](const IdentExpr &e) -> Expected<Value> {
      // `null` is a literal, not a scope variable (the lexer has no NULL
      // keyword, so it arrives as an identifier).  Used e.g. as a GOMP_task
      // call argument; lowers to a null pointer.
      if (e.name == "null") return makeNull();
      return scope.get(e.name);
    },
    [&](const StringExpr &e) -> Expected<Value> {
      return makeStr(e.value);
    },
    [&](const NumberExpr &e) -> Expected<Value> {
      return makeInt(e.value);
    },
    [&](const BoolExpr &e) -> Expected<Value> {
      return makeBool(e.value);
    },
    [&](const ListExpr &e) -> Expected<Value> {
      ListVal l;
      for (auto &v : e.values) {
        auto r = evalExpr(v, scope);
        if (!r) return r.takeError();
        l.items.push_back(std::make_shared<ValueBox>(ValueBox{std::move(*r)}));
      }
      return Value{l};
    },
    [&](const CallExpr &e) -> Expected<Value> {
      // ident(<flag>): the flag is a bare token (e.g. work_loop), not a scope
      // variable. Emit the symbolic ident reference "%ident:<flag>"; the pass
      // maps the flag to the ident_t.flags bitmask. Bare `ident` (an IdentExpr,
      // handled elsewhere) stays "%ident" / default flags.
      if (e.name == "ident") {
        std::string flag;
        if (e.args.size() == 1) {
          auto r = evalExprOrBare(e.args[0], scope);
          if (!r) return r.takeError();
          flag = valueToString(*r);
        }
        return makeStr(flag.empty() ? "%ident" : "%ident:" + flag);
      }
      std::vector<Value> args;
      for (auto &a : e.args) {
        auto r = evalExpr(a, scope);
        if (!r) return r.takeError();
        args.push_back(std::move(*r));
      }
      return evalBuiltin(e.name, std::move(args));
    }
  ), expr);
}

// Like evalExpr but if expr is an IdentExpr not in scope, return the name
// as a bare string (for patterns like `schedule == static`).
Expected<Value> evalExprOrBare(const Expr &expr, ScopeImpl &scope) {
  if (auto *id = std::get_if<IdentExpr>(&expr)) {
    if (!scope.has(id->name))
      return makeStr(id->name);
  }
  return evalExpr(expr, scope);
}

// ---- Predicate ------------------------------------------------------------

Expected<bool> evalPredicate(const Predicate &pred, ScopeImpl &scope) {
  return std::visit(llvm::makeVisitor(
    [&](const PredHas &p) -> Expected<bool> {
      if (!scope.has(p.name)) return false;
      auto v = scope.get(p.name);
      if (!v) return v.takeError();
      return !std::holds_alternative<NullVal>(*v);
    },
    [&](const PredEq &p) -> Expected<bool> {
      auto lhs = scope.get(p.name);
      if (!lhs) return lhs.takeError();
      auto rhs = evalExprOrBare(p.value, scope);
      if (!rhs) return rhs.takeError();
      return valuesEqual(*lhs, *rhs);
    },
    [&](const PredNe &p) -> Expected<bool> {
      auto lhs = scope.get(p.name);
      if (!lhs) return lhs.takeError();
      auto rhs = evalExprOrBare(p.value, scope);
      if (!rhs) return rhs.takeError();
      return !valuesEqual(*lhs, *rhs);
    },
    [&](const PredIdent &p) -> Expected<bool> {
      auto v = scope.get(p.name);
      if (!v) return v.takeError();
      return isTruthy(*v);
    },
    [&](const PredBool &p) -> Expected<bool> {
      return p.value;
    },
    [&](const PredNot &p) -> Expected<bool> {
      auto v = evalPredicate(*p.inner, scope);
      if (!v) return v.takeError();
      return !*v;
    },
    [&](const PredAnd &p) -> Expected<bool> {
      auto l = evalPredicate(*p.left, scope);
      if (!l) return l.takeError();
      if (!*l) return false;
      return evalPredicate(*p.right, scope);
    },
    [&](const PredOr &p) -> Expected<bool> {
      auto l = evalPredicate(*p.left, scope);
      if (!l) return l.takeError();
      if (*l) return true;
      return evalPredicate(*p.right, scope);
    }
  ), pred);
}

// ---- Action ---------------------------------------------------------------

Expected<std::string> resolveCalleeStr(const Expr &callee, ScopeImpl &scope) {
  auto v = evalExpr(callee, scope);
  if (!v) return v.takeError();
  return valueToString(*v);
}

Expected<PlanAction> evalAction(const Action &action, ScopeImpl &scope) {
  return std::visit(llvm::makeVisitor(
    [&](const EmitAction &a) -> Expected<PlanAction> {
      Value val;
      if (a.arg) {
        // `emit name(arg)`: the argument value rides the emit's value slot.
        auto v = evalExpr(*a.arg, scope);
        if (!v) return v.takeError();
        val = std::move(*v);
      } else if (scope.has(a.name)) {
        auto v = scope.get(a.name);
        if (!v) return v.takeError();
        val = std::move(*v);
      } else {
        val = makeStr(a.name);
      }
      return PlanAction{PlanEmit{a.name, std::move(val)}};
    },
    [&](const CallAction &a) -> Expected<PlanAction> {
      auto callee = resolveCalleeStr(a.callee, scope);
      if (!callee) return callee.takeError();
      std::vector<Value> args;
      for (auto &arg : a.args) {
        auto v = evalExpr(arg, scope);
        if (!v) return v.takeError();
        args.push_back(std::move(*v));
      }
      return PlanAction{PlanCall{std::move(*callee), std::move(args)}};
    }
  ), action);
}

// ---- Block (with when/otherwise chain semantics) --------------------------

Expected<std::vector<PlanAction>> evalBlock(const BlockDecl &block,
                                            ScopeImpl &parentScope) {
  ScopeImpl scope(&parentScope);
  std::vector<PlanAction> out;
  size_t i = 0, n = block.statements.size();

  while (i < n) {
    const auto &stmt = block.statements[i];

    if (auto *ls = std::get_if<LetStmt>(&stmt)) {
      auto v = evalExpr(ls->decl.expr, scope);
      if (!v) return v.takeError();
      scope.set(ls->decl.name, std::move(*v));
      i++;
      continue;
    }

    if (auto *lc = std::get_if<LetCallStmt>(&stmt)) {
      // Emit the call and bind <name> to a symbolic placeholder ("%name") so
      // later argument expressions referencing it resolve to that token; the
      // pass maps the token back to the call's SSA result.
      auto a = evalAction(Action{lc->call}, scope);
      if (!a) return a.takeError();
      auto &pc = std::get<PlanCall>(*a);
      pc.resultName = lc->name;
      out.push_back(std::move(*a));
      scope.set(lc->name, makeStr("%" + lc->name));
      i++;
      continue;
    }

    if (auto *as = std::get_if<ActionStmt>(&stmt)) {
      auto a = evalAction(as->action, scope);
      if (!a) return a.takeError();
      out.push_back(std::move(*a));
      i++;
      continue;
    }

    if (std::holds_alternative<WhenStmt>(stmt)) {
      // Collect the when/otherwise chain
      bool taken = false;
      while (i < n) {
        const auto &cur = block.statements[i];
        if (auto *ws = std::get_if<WhenStmt>(&cur)) {
          if (!taken) {
            auto cond = evalPredicate(ws->predicate, scope);
            if (!cond) return cond.takeError();
            if (*cond) {
              auto a = evalAction(ws->action, scope);
              if (!a) return a.takeError();
              out.push_back(std::move(*a));
              taken = true;
            }
          }
          i++;
        } else if (auto *os = std::get_if<OtherwiseStmt>(&cur)) {
          if (!taken) {
            auto a = evalAction(os->action, scope);
            if (!a) return a.takeError();
            out.push_back(std::move(*a));
            taken = true;
          }
          i++;
          break;
        } else {
          break;
        }
      }
      continue;
    }

    if (std::holds_alternative<OtherwiseStmt>(stmt)) {
      return make_error<StringError>(
        "standalone 'otherwise' without 'when'", inconvertibleErrorCode());
    }

    i++;
  }
  return out;
}

// ---- Construct selection --------------------------------------------------

Expected<const ConstructDecl *> selectConstruct(
    const RuntimeDecl &runtime,
    StringRef constructName,
    ScopeImpl &scope) {

  std::vector<const ConstructDecl *> candidates;
  for (auto &item : runtime.items) {
    if (auto *cd = std::get_if<ConstructDecl>(&item))
      if (constructName == cd->name)
        candidates.push_back(cd);
  }
  if (candidates.empty())
    return make_error<StringError>(
      "construct '" + constructName.str() + "' not found in runtime '" +
      runtime.name + "'", inconvertibleErrorCode());

  // Evaluate guards
  std::vector<const ConstructDecl *> matched;
  for (auto *c : candidates) {
    if (!c->guard) {
      matched.push_back(c);
    } else {
      auto v = evalPredicate(*c->guard, scope);
      if (!v) return v.takeError();
      if (*v) matched.push_back(c);
    }
  }

  if (matched.empty())
    return make_error<StringError>(
      "no construct variant matched for '" + constructName.str() + "'",
      inconvertibleErrorCode());

  if (matched.size() > 1) {
    // Prefer guarded over unguarded
    std::vector<const ConstructDecl *> guarded;
    for (auto *c : matched) if (c->guard) guarded.push_back(c);
    if (guarded.size() == 1) return guarded[0];
    if (guarded.size() > 1)
      return make_error<StringError>(
        "ambiguous construct selection for '" + constructName.str() + "'",
        inconvertibleErrorCode());
  }
  return matched[0];
}

} // anonymous namespace

// ===========================================================================
// Evaluator::buildPlan
// ===========================================================================

const RuntimeDecl *Evaluator::findRuntime(StringRef name) const {
  for (auto &rt : program.runtimes)
    if (name == rt.name) return &rt;
  return nullptr;
}

Expected<LoweringPlan> Evaluator::buildPlan(
    StringRef runtimeName,
    StringRef constructName,
    const llvm::StringMap<Value> &context) {

  const RuntimeDecl *runtime = findRuntime(runtimeName);
  if (!runtime)
    return make_error<StringError>(
      "runtime '" + runtimeName.str() + "' not found",
      inconvertibleErrorCode());

  // Root scope from context
  ScopeImpl root = buildRootScope(context);

  // Evaluate runtime-level lets
  for (auto &item : runtime->items) {
    if (auto *ld = std::get_if<LetDecl>(&item)) {
      auto v = evalExpr(ld->expr, root);
      if (!v) return v.takeError();
      root.set(ld->name, std::move(*v));
    }
  }

  // Select construct
  auto cdResult = selectConstruct(*runtime, constructName, root);
  if (!cdResult) return cdResult.takeError();
  const ConstructDecl *cd = *cdResult;

  LoweringPlan plan;
  plan.runtime   = runtime->name;
  plan.construct = cd->name;

  // Evaluate runtime-level properties into the plan. They are shared by every
  // construct of the runtime; a construct-level property of the same name
  // overrides (the construct loop below runs afterwards).
  for (auto &item : runtime->items) {
    if (auto *pd = std::get_if<PropertyDecl>(&item)) {
      auto v = evalExpr(pd->expr, root);
      if (!v) return v.takeError();
      plan.properties[pd->name] = std::move(*v);
    }
  }

  // Construct scope inherits runtime scope
  ScopeImpl cscope(&root);

  for (auto &item : cd->items) {
    if (auto *ld = std::get_if<LetDecl>(&item)) {
      auto v = evalExpr(ld->expr, cscope);
      if (!v) return v.takeError();
      cscope.set(ld->name, std::move(*v));

    } else if (auto *pd = std::get_if<PropertyDecl>(&item)) {
      auto v = evalExpr(pd->expr, cscope);
      if (!v) return v.takeError();
      plan.properties[pd->name] = std::move(*v);

    } else if (auto *bd = std::get_if<BlockDecl>(&item)) {
      auto actions = evalBlock(*bd, cscope);
      if (!actions) return actions.takeError();
      if (bd->name == "pre")
        for (auto &a : *actions) plan.pre.push_back(std::move(a));
      else if (bd->name == "invoke")
        for (auto &a : *actions) plan.invoke.push_back(std::move(a));
      else if (bd->name == "post")
        for (auto &a : *actions) plan.post.push_back(std::move(a));
      else
        return make_error<StringError>(
          "unknown block: " + bd->name, inconvertibleErrorCode());
    }
  }

  return plan;
}
