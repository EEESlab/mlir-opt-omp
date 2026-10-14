// Turns a parsed Program plus a runtime/construct/context into a LoweringPlan.

#pragma once

#include "OmpLowering/DSL/DSLParser.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace dsl {

// --- Values ---
// ListVal boxes its items because Value is a variant that includes ListVal.

struct ValueBox;  // forward declaration

struct NullVal {};
struct IntVal   { int value; };
struct BoolVal  { bool value; };
struct StrVal   { std::string value; };
struct ListVal  { std::vector<std::shared_ptr<ValueBox>> items; };

using Value = std::variant<NullVal, IntVal, BoolVal, StrVal, ListVal>;

struct ValueBox { Value v; };

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

inline std::vector<Value> listItems(const ListVal &l) {
  std::vector<Value> out;
  out.reserve(l.items.size());
  for (auto &p : l.items) out.push_back(p->v);
  return out;
}

bool isTruthy(const Value &v);
std::string valueToString(const Value &v);

// --- Plan model ---

struct PlanEmit {
  std::string name;
  Value value;
};

struct PlanCall {
  std::string callee;
  std::vector<Value> args;
  std::string resultName;  // non-empty if bound by `let = call`
};

// A branch on a run-time value.  `when`/`otherwise` are decided during
// evaluation and collapse; a PlanBranch survives into real control flow.
struct PlanActionBox;

struct PlanBranch {
  Value cond;
  std::vector<std::shared_ptr<PlanActionBox>> ifTrue;
  std::vector<std::shared_ptr<PlanActionBox>> ifFalse;
};

using PlanAction = std::variant<PlanEmit, PlanCall, PlanBranch>;

struct PlanActionBox { PlanAction action; };

struct LoweringPlan {
  std::string runtime;
  std::string construct;
  std::map<std::string, Value> properties;
  std::vector<PlanAction> pre;
  std::vector<PlanAction> invoke;
  std::vector<PlanAction> post;
  // Chunked work-sharing loops: `nextChunk` runs before each turn and fills
  // the bound slots, a falsy result ends the loop; `firstChunk` is for
  // runtimes whose opening call differs (libgomp), empty where it does not
  // (iomp).  A non-empty `nextChunk` is what marks a loop as chunked — the
  // passes never inspect the schedule kind themselves.
  std::vector<PlanAction> firstChunk;
  std::vector<PlanAction> nextChunk;
};

// --- Evaluator ---

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
