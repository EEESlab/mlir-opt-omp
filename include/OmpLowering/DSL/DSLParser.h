// AST node types and the entry point that parses DSL source into a Program.

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace dsl {

// --- Expr ---

struct IdentExpr;
struct StringExpr;
struct NumberExpr;
struct BoolExpr;
struct CallExpr;
struct ListExpr;

using Expr = std::variant<
  IdentExpr, StringExpr, NumberExpr, BoolExpr, CallExpr, ListExpr>;

struct IdentExpr  { std::string name; };
struct StringExpr { std::string value; };
struct NumberExpr { int value; };
struct BoolExpr   { bool value; };
struct CallExpr   { std::string name; std::vector<Expr> args; };
struct ListExpr   { std::vector<Expr> values; };

// --- Predicate ---

struct PredHas;
struct PredEq;
struct PredNe;
struct PredIdent;
struct PredBool;
struct PredNot;
struct PredAnd;
struct PredOr;

using Predicate = std::variant<
  PredHas, PredEq, PredNe, PredIdent, PredBool,
  PredNot, PredAnd, PredOr>;

struct PredHas   { std::string name; };
struct PredEq    { std::string name; Expr value; };
struct PredNe    { std::string name; Expr value; };
struct PredIdent { std::string name; };
struct PredBool  { bool value; };
struct PredNot   { std::shared_ptr<Predicate> inner; };
struct PredAnd   { std::shared_ptr<Predicate> left, right; };
struct PredOr    { std::shared_ptr<Predicate> left, right; };

// --- Action ---

struct EmitAction { std::string name; std::optional<Expr> arg; };
struct CallAction { Expr callee; std::vector<Expr> args; };
using Action = std::variant<EmitAction, CallAction>;

// --- Statement ---

struct LetDecl;
struct ActionStmt;
struct WhenStmt;
struct OtherwiseStmt;
struct LetStmt;
struct LetCallStmt;
struct BranchStmt;

struct LetDecl { std::string name; Expr expr; };

using Statement =
  std::variant<ActionStmt, WhenStmt, OtherwiseStmt, LetStmt, LetCallStmt,
               BranchStmt>;

struct ActionStmt    { Action action; };
struct WhenStmt      { Predicate predicate; Action action; };
struct OtherwiseStmt { Action action; };
// Unlike `when`, which is settled during evaluation, this condition is a
// run-time value and becomes a real branch in the emitted IR.  Arms hold
// statements so the two kinds of choice can nest.  A null condition means the
// clause is absent: the true arm is emitted inline and no branch survives.
struct BranchStmt {
  Expr condition;
  std::vector<Statement> ifTrue;
  std::vector<Statement> ifFalse;
};
struct LetStmt       { LetDecl decl; };
// Binds <name> to the call's SSA result for later statements.
struct LetCallStmt   { std::string name; CallAction call; };

// --- Construct items ---

struct PropertyDecl { std::string name; Expr expr; };
struct BlockDecl    { std::string name; std::vector<Statement> statements; };

using ConstructItem = std::variant<LetDecl, PropertyDecl, BlockDecl>;

// --- Top-level ---

struct ConstructDecl {
  std::string name;
  std::optional<Predicate> guard;
  std::vector<ConstructItem> items;
};

using RuntimeItem = std::variant<LetDecl, PropertyDecl, ConstructDecl>;

struct RuntimeDecl {
  std::string name;
  std::vector<RuntimeItem> items;
};

struct Program {
  std::vector<RuntimeDecl> runtimes;
};

// --- Entry point ---

llvm::Expected<Program> parse(llvm::StringRef source);

} // namespace dsl
