// Lexer and parser for the rules DSL.

#include "OmpLowering/DSL/DSLParser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <string>
#include <unordered_map>
#include <vector>

using namespace dsl;
using llvm::Expected;
using llvm::make_error;
using llvm::StringError;
using llvm::inconvertibleErrorCode;

// --- Token ---

enum class TK {
  // Keywords
  // pre/invoke/post are deliberately NOT keywords: a block is any identifier
  // followed by '{', which lets a construct declare first_chunk / next_chunk
  // without a token of their own.  The evaluator rejects a name it cannot use.
  RUNTIME, CONSTRUCT, WHEN, OTHERWISE,
  LET, CALL, EMIT, HAS, AND, OR, NOT, TRUE, FALSE, BRANCH,
  // Punctuation
  LBRACE, RBRACE, LPAREN, RPAREN, LBRACKET, RBRACKET,
  SEMI, COMMA, EQUALS, ARROW, EQEQ, NEQ,
  // Values
  IDENT, STRING, NUMBER,
  // End
  END
};

struct Token {
  TK kind;
  std::string value;
  int line, col;
};

// --- Lexer ---

class Lexer {
  std::string src;
  size_t pos = 0;
  int line = 1, col = 1;

  char peek(int offset = 0) const {
    size_t i = pos + offset;
    return i < src.size() ? src[i] : '\0';
  }

  char advance() {
    char c = src[pos++];
    if (c == '\n') { line++; col = 1; } else col++;
    return c;
  }

  Token make(TK k, std::string v) { return {k, std::move(v), line, col}; }

  void skipLineComment() {
    while (pos < src.size() && src[pos] != '\n') pos++;
  }

  void skipBlockComment() {
    pos += 2; // skip /*
    while (pos + 1 < src.size()) {
      if (src[pos] == '*' && src[pos+1] == '/') { pos += 2; return; }
      if (src[pos] == '\n') { line++; col = 1; }
      pos++;
    }
  }

  Expected<Token> lexString() {
    int sl = line, sc = col;
    advance(); // opening quote
    std::string s;
    while (pos < src.size()) {
      char c = src[pos];
      if (c == '"') { advance(); return Token{TK::STRING, s, sl, sc}; }
      if (c == '\\') {
        advance();
        if (pos >= src.size())
          return make_error<StringError>("unterminated string", inconvertibleErrorCode());
        char e = advance();
        switch (e) {
          case '"':  s += '"';  break;
          case '\\': s += '\\'; break;
          case 'n':  s += '\n'; break;
          case 't':  s += '\t'; break;
          default:
            return make_error<StringError>("bad escape", inconvertibleErrorCode());
        }
        continue;
      }
      s += advance();
    }
    return make_error<StringError>("unterminated string", inconvertibleErrorCode());
  }

  Token lexNumber() {
    std::string s;
    int sl = line, sc = col;
    while (pos < src.size() && std::isdigit(src[pos])) s += advance();
    return {TK::NUMBER, s, sl, sc};
  }

  Token lexIdentOrKeyword() {
    std::string s;
    int sl = line, sc = col;
    while (pos < src.size() && (std::isalnum(src[pos]) || src[pos] == '_'))
      s += advance();

    static const std::unordered_map<std::string, TK> kw = {
      {"runtime",   TK::RUNTIME},   {"construct", TK::CONSTRUCT},
      {"when",      TK::WHEN},      {"otherwise", TK::OTHERWISE},
      {"let",       TK::LET},
      {"call",      TK::CALL},      {"emit",      TK::EMIT},
      {"has",       TK::HAS},       {"and",       TK::AND},
      {"or",        TK::OR},        {"not",       TK::NOT},
      {"true",      TK::TRUE},      {"false",     TK::FALSE},
      {"branch",    TK::BRANCH},
    };
    auto it = kw.find(s);
    TK k = (it != kw.end()) ? it->second : TK::IDENT;
    return {k, s, sl, sc};
  }

public:
  explicit Lexer(std::string s) : src(std::move(s)) {}

  Expected<std::vector<Token>> tokenize() {
    std::vector<Token> tokens;
    while (pos < src.size()) {
      char c = src[pos];
      if (c == ' ' || c == '\t' || c == '\r') { advance(); continue; }
      if (c == '\n') { advance(); continue; }
      if (c == '/' && peek(1) == '/') { skipLineComment(); continue; }
      if (c == '/' && peek(1) == '*') { skipBlockComment(); continue; }

      if (c == '{') { tokens.push_back(make(TK::LBRACE,   "{")); advance(); continue; }
      if (c == '}') { tokens.push_back(make(TK::RBRACE,   "}")); advance(); continue; }
      if (c == '(') { tokens.push_back(make(TK::LPAREN,   "(")); advance(); continue; }
      if (c == ')') { tokens.push_back(make(TK::RPAREN,   ")")); advance(); continue; }
      if (c == '[') { tokens.push_back(make(TK::LBRACKET, "[")); advance(); continue; }
      if (c == ']') { tokens.push_back(make(TK::RBRACKET, "]")); advance(); continue; }
      if (c == ';') { tokens.push_back(make(TK::SEMI,     ";")); advance(); continue; }
      if (c == ',') { tokens.push_back(make(TK::COMMA,    ",")); advance(); continue; }

      if (c == '=' && peek(1) == '>') {
        tokens.push_back(make(TK::ARROW, "=>")); advance(); advance(); continue;
      }
      if (c == '=' && peek(1) == '=') {
        tokens.push_back(make(TK::EQEQ,  "==")); advance(); advance(); continue;
      }
      if (c == '!' && peek(1) == '=') {
        tokens.push_back(make(TK::NEQ,   "!=")); advance(); advance(); continue;
      }
      if (c == '=') {
        tokens.push_back(make(TK::EQUALS, "=")); advance(); continue;
      }
      if (c == '"') {
        auto t = lexString();
        if (!t) return t.takeError();
        tokens.push_back(std::move(*t));
        continue;
      }
      if (std::isdigit(c)) { tokens.push_back(lexNumber()); continue; }
      if (std::isalpha(c) || c == '_') {
        tokens.push_back(lexIdentOrKeyword()); continue;
      }
      return make_error<StringError>(
        std::string("unexpected char '") + c + "'",
        inconvertibleErrorCode());
    }
    tokens.push_back({TK::END, "", line, col});
    return tokens;
  }
};

// --- Parser ---

class Parser {
  std::vector<Token> toks;
  size_t pos = 0;

  const Token &cur() const { return toks[pos]; }
  bool at(TK k) const { return cur().kind == k; }
  // One-token lookahead, for where the next token decides how to read this one
  // (an identifier opening a block vs a property).
  bool at(TK k, size_t off) const {
    return pos + off < toks.size() && toks[pos + off].kind == k;
  }

  Expected<Token> expect(TK k) {
    if (!at(k))
      return make_error<StringError>(
        "expected token " + std::to_string((int)k) +
        " got '" + cur().value + "' at line " + std::to_string(cur().line),
        inconvertibleErrorCode());
    return toks[pos++];
  }

  Token advance() { return toks[pos++]; }

  // ---- Expr ---------------------------------------------------------------

  Expected<Expr> parseExpr() { return parsePrimaryExpr(); }

  Expected<Expr> parsePrimaryExpr() {
    if (at(TK::IDENT)) {
      auto name = advance().value;
      if (at(TK::LPAREN)) {
        advance();
        std::vector<Expr> args;
        if (!at(TK::RPAREN)) {
          auto a = parseArgList();
          if (!a) return a.takeError();
          args = std::move(*a);
        }
        if (auto e = expect(TK::RPAREN); !e) return e.takeError();
        return Expr{CallExpr{name, std::move(args)}};
      }
      return Expr{IdentExpr{name}};
    }
    if (at(TK::STRING)) return Expr{StringExpr{advance().value}};
    if (at(TK::NUMBER)) return Expr{NumberExpr{std::stoi(advance().value)}};
    if (at(TK::TRUE))  { advance(); return Expr{BoolExpr{true}}; }
    if (at(TK::FALSE)) { advance(); return Expr{BoolExpr{false}}; }
    if (at(TK::LBRACKET)) {
      advance();
      std::vector<Expr> vals;
      if (!at(TK::RBRACKET)) {
        auto a = parseArgList();
        if (!a) return a.takeError();
        vals = std::move(*a);
      }
      if (auto e = expect(TK::RBRACKET); !e) return e.takeError();
      return Expr{ListExpr{std::move(vals)}};
    }
    if (at(TK::LPAREN)) {
      advance();
      auto e = parseExpr();
      if (!e) return e.takeError();
      if (auto r = expect(TK::RPAREN); !r) return r.takeError();
      return e;
    }
    return make_error<StringError>(
      "invalid expression at line " + std::to_string(cur().line),
      inconvertibleErrorCode());
  }

  Expected<std::vector<Expr>> parseArgList() {
    std::vector<Expr> args;
    auto first = parseExpr();
    if (!first) return first.takeError();
    args.push_back(std::move(*first));
    while (at(TK::COMMA)) {
      advance();
      auto e = parseExpr();
      if (!e) return e.takeError();
      args.push_back(std::move(*e));
    }
    return args;
  }

  // ---- Predicate ----------------------------------------------------------

  Expected<Predicate> parsePredicate()    { return parsePredicateOr(); }

  Expected<Predicate> parsePredicateOr() {
    auto left = parsePredicateAnd();
    if (!left) return left.takeError();
    while (at(TK::OR)) {
      advance();
      auto right = parsePredicateAnd();
      if (!right) return right.takeError();
      left = Predicate{PredOr{
        std::make_shared<Predicate>(std::move(*left)),
        std::make_shared<Predicate>(std::move(*right))}};
    }
    return left;
  }

  Expected<Predicate> parsePredicateAnd() {
    auto left = parsePredicateNot();
    if (!left) return left.takeError();
    while (at(TK::AND)) {
      advance();
      auto right = parsePredicateNot();
      if (!right) return right.takeError();
      left = Predicate{PredAnd{
        std::make_shared<Predicate>(std::move(*left)),
        std::make_shared<Predicate>(std::move(*right))}};
    }
    return left;
  }

  Expected<Predicate> parsePredicateNot() {
    if (at(TK::NOT)) {
      advance();
      auto inner = parsePredicateNot();
      if (!inner) return inner.takeError();
      return Predicate{PredNot{std::make_shared<Predicate>(std::move(*inner))}};
    }
    return parsePredicateAtom();
  }

  Expected<Predicate> parsePredicateAtom() {
    if (at(TK::HAS)) {
      advance();
      if (auto e = expect(TK::LPAREN); !e) return e.takeError();
      auto name = cur().value;
      if (auto e = expect(TK::IDENT); !e) return e.takeError();
      if (auto e = expect(TK::RPAREN); !e) return e.takeError();
      return Predicate{PredHas{name}};
    }
    if (at(TK::LPAREN)) {
      advance();
      auto p = parsePredicate();
      if (!p) return p.takeError();
      if (auto e = expect(TK::RPAREN); !e) return e.takeError();
      return p;
    }
    if (at(TK::TRUE))  { advance(); return Predicate{PredBool{true}}; }
    if (at(TK::FALSE)) { advance(); return Predicate{PredBool{false}}; }
    if (at(TK::IDENT)) {
      auto name = advance().value;
      if (at(TK::EQEQ)) {
        advance();
        auto v = parseExpr();
        if (!v) return v.takeError();
        return Predicate{PredEq{name, std::move(*v)}};
      }
      if (at(TK::NEQ)) {
        advance();
        auto v = parseExpr();
        if (!v) return v.takeError();
        return Predicate{PredNe{name, std::move(*v)}};
      }
      return Predicate{PredIdent{name}};
    }
    return make_error<StringError>(
      "invalid predicate at line " + std::to_string(cur().line),
      inconvertibleErrorCode());
  }

  // ---- Action -------------------------------------------------------------

  Expected<Action> parseAction() {
    if (at(TK::CALL)) {
      advance();
      Expr callee;
      if (at(TK::IDENT))        callee = IdentExpr{advance().value};
      else if (at(TK::STRING))  callee = StringExpr{advance().value};
      else return make_error<StringError>("expected callee", inconvertibleErrorCode());

      if (auto e = expect(TK::LPAREN); !e) return e.takeError();
      std::vector<Expr> args;
      if (!at(TK::RPAREN)) {
        auto a = parseArgList();
        if (!a) return a.takeError();
        args = std::move(*a);
      }
      if (auto e = expect(TK::RPAREN); !e) return e.takeError();
      return Action{CallAction{std::move(callee), std::move(args)}};
    }
    if (at(TK::EMIT)) {
      advance();
      auto name = cur().value;
      if (auto e = expect(TK::IDENT); !e) return e.takeError();
      // Optional single parenthesized argument, e.g. `emit populate_shareds(task)`.
      std::optional<Expr> arg;
      if (at(TK::LPAREN)) {
        advance();
        auto a = parseExpr();
        if (!a) return a.takeError();
        arg = std::move(*a);
        if (auto e = expect(TK::RPAREN); !e) return e.takeError();
      }
      return Action{EmitAction{name, std::move(arg)}};
    }
    return make_error<StringError>(
      "expected 'call' or 'emit' at line " + std::to_string(cur().line),
      inconvertibleErrorCode());
  }

  // ---- Statement ----------------------------------------------------------

  // One arm of a `branch`: a single statement, or a braced sequence of them.
  Expected<std::vector<Statement>> parseBranchArm() {
    std::vector<Statement> stmts;
    if (at(TK::LBRACE)) {
      advance();
      while (!at(TK::RBRACE) && !at(TK::END)) {
        auto st = parseStatement();
        if (!st) return st.takeError();
        stmts.push_back(std::move(*st));
      }
      if (auto e = expect(TK::RBRACE); !e) return e.takeError();
      return stmts;
    }
    auto st = parseStatement();
    if (!st) return st.takeError();
    stmts.push_back(std::move(*st));
    return stmts;
  }

  Expected<Statement> parseStatement() {
    if (at(TK::BRANCH)) {
      advance();
      auto cond = parseExpr();
      if (!cond) return cond.takeError();
      if (auto e = expect(TK::LBRACE); !e) return e.takeError();

      BranchStmt br{std::move(*cond), {}, {}};
      bool sawTrue = false, sawFalse = false;
      while (at(TK::TRUE) || at(TK::FALSE)) {
        bool isTrue = at(TK::TRUE);
        int line = cur().line;
        advance();
        if (auto e = expect(TK::ARROW); !e) return e.takeError();
        auto arm = parseBranchArm();
        if (!arm) return arm.takeError();
        if (isTrue) {
          if (sawTrue)
            return make_error<StringError>(
              "duplicate 'true' arm in branch at line " + std::to_string(line),
              inconvertibleErrorCode());
          sawTrue = true;
          br.ifTrue = std::move(*arm);
        } else {
          if (sawFalse)
            return make_error<StringError>(
              "duplicate 'false' arm in branch at line " + std::to_string(line),
              inconvertibleErrorCode());
          sawFalse = true;
          br.ifFalse = std::move(*arm);
        }
      }
      if (auto e = expect(TK::RBRACE); !e) return e.takeError();
      if (!sawTrue && !sawFalse)
        return make_error<StringError>(
          "branch needs at least a 'true' or a 'false' arm at line " +
            std::to_string(cur().line),
          inconvertibleErrorCode());
      return Statement{std::move(br)};
    }
    if (at(TK::WHEN)) {
      advance();
      auto pred = parsePredicate();
      if (!pred) return pred.takeError();
      if (auto e = expect(TK::ARROW); !e) return e.takeError();
      auto act = parseAction();
      if (!act) return act.takeError();
      if (auto e = expect(TK::SEMI); !e) return e.takeError();
      return Statement{WhenStmt{std::move(*pred), std::move(*act)}};
    }
    if (at(TK::OTHERWISE)) {
      advance();
      if (auto e = expect(TK::ARROW); !e) return e.takeError();
      auto act = parseAction();
      if (!act) return act.takeError();
      if (auto e = expect(TK::SEMI); !e) return e.takeError();
      return Statement{OtherwiseStmt{std::move(*act)}};
    }
    if (at(TK::LET)) {
      advance();
      auto name = cur().value;
      if (auto e = expect(TK::IDENT); !e) return e.takeError();
      if (auto e = expect(TK::EQUALS); !e) return e.takeError();
      // `let <name> = call "<callee>"(...)` binds the call's SSA result; any
      // other RHS is an expression binding.
      if (at(TK::CALL)) {
        auto act = parseAction();
        if (!act) return act.takeError();
        auto *ca = std::get_if<CallAction>(&*act);
        if (!ca)
          return make_error<StringError>(
            "expected a call after 'let " + name + " =' at line " +
              std::to_string(cur().line),
            inconvertibleErrorCode());
        if (auto e = expect(TK::SEMI); !e) return e.takeError();
        return Statement{LetCallStmt{name, std::move(*ca)}};
      }
      auto expr = parseExpr();
      if (!expr) return expr.takeError();
      if (auto e = expect(TK::SEMI); !e) return e.takeError();
      return Statement{LetStmt{LetDecl{name, std::move(*expr)}}};
    }
    auto act = parseAction();
    if (!act) return act.takeError();
    if (auto e = expect(TK::SEMI); !e) return e.takeError();
    return Statement{ActionStmt{std::move(*act)}};
  }

  // ---- Construct items ----------------------------------------------------

  // `<name> { <statements> }`.  Any identifier will do — the name says *when*
  // the statements run, and which names mean something is the evaluator's
  // business, not the parser's.
  Expected<BlockDecl> parseBlock() {
    std::string name = cur().value;
    if (auto e = expect(TK::IDENT); !e) return e.takeError();

    if (auto e = expect(TK::LBRACE); !e) return e.takeError();
    std::vector<Statement> stmts;
    while (!at(TK::RBRACE) && !at(TK::END)) {
      auto s = parseStatement();
      if (!s) return s.takeError();
      stmts.push_back(std::move(*s));
    }
    if (auto e = expect(TK::RBRACE); !e) return e.takeError();
    return BlockDecl{name, std::move(stmts)};
  }

  Expected<LetDecl> parseLetDecl() {
    if (auto e = expect(TK::LET); !e) return e.takeError();
    auto name = cur().value;
    if (auto e = expect(TK::IDENT); !e) return e.takeError();
    if (auto e = expect(TK::EQUALS); !e) return e.takeError();
    auto expr = parseExpr();
    if (!expr) return expr.takeError();
    if (auto e = expect(TK::SEMI); !e) return e.takeError();
    return LetDecl{name, std::move(*expr)};
  }

  Expected<PropertyDecl> parsePropertyDecl() {
    auto name = cur().value;
    if (auto e = expect(TK::IDENT); !e) return e.takeError();
    if (auto e = expect(TK::EQUALS); !e) return e.takeError();
    auto expr = parseExpr();
    if (!expr) return expr.takeError();
    if (auto e = expect(TK::SEMI); !e) return e.takeError();
    return PropertyDecl{name, std::move(*expr)};
  }

  Expected<ConstructDecl> parseConstructDecl() {
    if (auto e = expect(TK::CONSTRUCT); !e) return e.takeError();
    auto name = cur().value;
    if (auto e = expect(TK::IDENT); !e) return e.takeError();

    std::optional<Predicate> guard;
    if (at(TK::WHEN)) {
      advance();
      auto p = parsePredicate();
      if (!p) return p.takeError();
      guard = std::move(*p);
    }

    if (auto e = expect(TK::LBRACE); !e) return e.takeError();
    std::vector<ConstructItem> items;
    while (!at(TK::RBRACE) && !at(TK::END)) {
      if (at(TK::LET)) {
        auto ld = parseLetDecl();
        if (!ld) return ld.takeError();
        items.push_back(std::move(*ld));
      } else if (at(TK::IDENT)) {
        // The token after the name tells the two apart: '{' opens a block,
        // '=' a property.
        if (at(TK::LBRACE, 1)) {
          auto bd = parseBlock();
          if (!bd) return bd.takeError();
          items.push_back(std::move(*bd));
        } else {
          auto pd = parsePropertyDecl();
          if (!pd) return pd.takeError();
          items.push_back(std::move(*pd));
        }
      } else {
        return make_error<StringError>(
          "unexpected token in construct at line " + std::to_string(cur().line),
          inconvertibleErrorCode());
      }
    }
    if (auto e = expect(TK::RBRACE); !e) return e.takeError();
    return ConstructDecl{name, std::move(guard), std::move(items)};
  }

  Expected<RuntimeDecl> parseRuntimeDecl() {
    if (auto e = expect(TK::RUNTIME); !e) return e.takeError();
    auto name = cur().value;
    if (auto e = expect(TK::IDENT); !e) return e.takeError();
    if (auto e = expect(TK::LBRACE); !e) return e.takeError();

    std::vector<RuntimeItem> items;
    while (!at(TK::RBRACE) && !at(TK::END)) {
      if (at(TK::LET)) {
        auto ld = parseLetDecl();
        if (!ld) return ld.takeError();
        items.push_back(std::move(*ld));
      } else if (at(TK::CONSTRUCT)) {
        auto cd = parseConstructDecl();
        if (!cd) return cd.takeError();
        items.push_back(std::move(*cd));
      } else if (at(TK::IDENT)) {
        auto pd = parsePropertyDecl();
        if (!pd) return pd.takeError();
        items.push_back(std::move(*pd));
      } else {
        return make_error<StringError>(
          "unexpected token in runtime at line " + std::to_string(cur().line),
          inconvertibleErrorCode());
      }
    }
    if (auto e = expect(TK::RBRACE); !e) return e.takeError();
    return RuntimeDecl{name, std::move(items)};
  }

public:
  explicit Parser(std::vector<Token> t) : toks(std::move(t)) {}

  Expected<Program> parseProgram() {
    Program prog;
    while (!at(TK::END)) {
      auto rt = parseRuntimeDecl();
      if (!rt) return rt.takeError();
      prog.runtimes.push_back(std::move(*rt));
    }
    return prog;
  }
};

// --- Entry point ---

namespace dsl {

llvm::Expected<Program> parse(llvm::StringRef source) {
  Lexer lexer(source.str());
  auto tokens = lexer.tokenize();
  if (!tokens) return tokens.takeError();
  Parser parser(std::move(*tokens));
  return parser.parseProgram();
}

} // namespace dsl
