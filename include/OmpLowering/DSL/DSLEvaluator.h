// DSLEvaluator.h
//
// C++ port of evaluator.py.
// Given a parsed dsl::Program, a runtime name, a construct name, and a
// context map, produces a LoweringPlan describing the sequence of runtime
// calls to emit.

#pragma once

#include "OmpLowering/DSL/DSLParser.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace dsl {

// ===========================================================================
// Value type used at evaluation time
// ===========================================================================
// We use shared_ptr<ValueBox> in ListVal to break the circular dependency
// that would arise if ListVal held std::vector<Value> directly, since Value
// is a variant that includes ListVal.

struct ValueBox;  // forward declaration

struct NullVal {};
struct IntVal   { int value; };
struct BoolVal  { bool value; };
struct StrVal   { std::string value; };
struct ListVal  { std::vector<std::shared_ptr<ValueBox>> items; };

using Value = std::variant<NullVal, IntVal, BoolVal, StrVal, ListVal>;

struct ValueBox { Value v; };

// Helpers
inline Value makeNull() { return NullVal{}; }
inline Value makeInt(int v) { return IntVal{v}; }
inline Value makeBool(bool v) { return BoolVal{v}; }
inline Value makeStr(std::string v) { return StrVal{std::move(v)}; }

inline Value makeList(std::vector<Value> items) {
  ListVal l;
  for (auto &x : items)
    l.items.push_back(std::make_shared<ValueBox>(ValueBox{std::move(x)}));
  return l;
}

// Unwrap a ListVal into a plain std::vector<Value> for evaluation.
inline std::vector<Value> listItems(const ListVal &l) {
  std::vector<Value> out;
  out.reserve(l.items.size());
  for (auto &p : l.items) out.push_back(p->v);
  return out;
}

bool isTruthy(const Value &v);
std::string valueToString(const Value &v);

// ===========================================================================
// Plan model  (mirrors Python PlanAction / LoweringPlan)
// ===========================================================================

struct PlanEmit {
  std::string name;
  Value value;
};

struct PlanCall {
  std::string callee;   // resolved to a string (function name)
  std::vector<Value> args;
};

using PlanAction = std::variant<PlanEmit, PlanCall>;

struct LoweringPlan {
  std::string runtime;
  std::string construct;
  std::map<std::string, Value> properties;
  std::vector<PlanAction> pre;
  std::vector<PlanAction> invoke;
  std::vector<PlanAction> post;
};

// ===========================================================================
// Evaluator
// ===========================================================================

class Evaluator {
  const Program &program;
  const RuntimeDecl *findRuntime(llvm::StringRef name) const;

public:
  explicit Evaluator(const Program &p) : program(p) {}

  llvm::Expected<LoweringPlan> buildPlan(
      llvm::StringRef runtimeName,
      llvm::StringRef constructName,
      const llvm::StringMap<Value> &context);
};

} // namespace dsl
