// DSLParser.h
//
// C++ port of parser.py.
// Provides:
//   - AST node types (Program, RuntimeDecl, ConstructDecl, …)
//   - Lexer / Parser: parse a DSL source string into a Program AST
//
// Usage:
//   auto program = dsl::parse(source);   // returns dsl::Program
//   // then pass to dsl::Evaluator

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace dsl {

// ===========================================================================
// Expr
// ===========================================================================

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

// ===========================================================================
// Predicate
// ===========================================================================

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

// ===========================================================================
// Action
// ===========================================================================

struct EmitAction { std::string name; std::optional<Expr> arg; };
struct CallAction { Expr callee; std::vector<Expr> args; };
using Action = std::variant<EmitAction, CallAction>;

// ===========================================================================
// Statement
// ===========================================================================

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
// `branch <expr> { true => <arm> false => <arm> }`, where an arm is one
// statement or a braced sequence of them.  Unlike `when`, whose predicate is
// decided while evaluating the rules, this condition is a value known only at
// run time, so it survives evaluation and becomes a real branch in the emitted
// IR.  Either arm may be empty.
//
// Arms hold statements, not just actions, so the two kinds of choice can nest:
// a `when has(num_threads)` inside an arm is still settled during evaluation
// and collapses, while the branch around it does not.
//
// A condition that evaluates to null means the clause is absent, and there is
// nothing to branch on: the true arm is emitted inline and no branch survives.
struct BranchStmt {
  Expr condition;
  std::vector<Statement> ifTrue;
  std::vector<Statement> ifFalse;
};
struct LetStmt       { LetDecl decl; };
// `let <name> = call "<callee>"(<args>);` — binds <name> to the call's SSA
// result so later statements can reference it (see docs/dsl-result-binding-proposal.md).
struct LetCallStmt   { std::string name; CallAction call; };

// ===========================================================================
// Construct items
// ===========================================================================

struct PropertyDecl { std::string name; Expr expr; };
struct BlockDecl    { std::string name; std::vector<Statement> statements; };

using ConstructItem = std::variant<LetDecl, PropertyDecl, BlockDecl>;

// ===========================================================================
// Top-level
// ===========================================================================

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

// ===========================================================================
// Entry point
// ===========================================================================

llvm::Expected<Program> parse(llvm::StringRef source);

} // namespace dsl
