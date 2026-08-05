// OmpToOmpLowerPass.cpp
//
// Converts omp.parallel / omp.task / omp.barrier / omp.taskwait ops to omp_lower.construct
// ops by evaluating the user-provided DSL file.  Wsloops are left in place
// (nested inside their parallel) and lowered later by OmpOutliningPass.

#include "OmpLowering/Transforms/OmpToOmpLowerPass.h"
#include "OmpLowering/IR/OmpLoweringOps.h"
#include "OmpLowering/DSL/DSLParser.h"
#include "OmpLowering/DSL/DSLEvaluator.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::omp_lower;

namespace {

// ===========================================================================
// dsl::Value -> MLIR Attribute conversion
// ===========================================================================

Attribute dslValueToAttr(const dsl::Value &v, MLIRContext *ctx) {
  return std::visit(llvm::makeVisitor(
    [&](const dsl::NullVal &)  -> Attribute {
      return StringAttr::get(ctx, "null");
    },
    [&](const dsl::BoolVal &b) -> Attribute {
      return BoolAttr::get(ctx, b.value);
    },
    [&](const dsl::IntVal &i)  -> Attribute {
      return IntegerAttr::get(IntegerType::get(ctx, 32), i.value);
    },
    [&](const dsl::StrVal &s)  -> Attribute {
      return StringAttr::get(ctx, s.value);
    },
    [&](const dsl::ListVal &l) -> Attribute {
      SmallVector<Attribute> attrs;
      for (auto &p : l.items)
        attrs.push_back(dslValueToAttr(p->v, ctx));
      return ArrayAttr::get(ctx, attrs);
    }
  ), v);
}

Attribute planActionToAttr(const dsl::PlanAction &action, MLIRContext *ctx) {
  return std::visit(llvm::makeVisitor(
    [&](const dsl::PlanEmit &e) -> Attribute {
      return PlanEmitAttr::get(
        ctx,
        StringAttr::get(ctx, e.name),
        dslValueToAttr(e.value, ctx));
    },
    [&](const dsl::PlanCall &c) -> Attribute {
      SmallVector<Attribute> argAttrs;
      for (auto &a : c.args)
        argAttrs.push_back(dslValueToAttr(a, ctx));
      return PlanCallAttr::get(
        ctx,
        StringAttr::get(ctx, c.callee),
        ArrayAttr::get(ctx, argAttrs),
        c.resultName.empty() ? StringAttr() : StringAttr::get(ctx, c.resultName));
    }
  ), action);
}

void emitConstructOp(const dsl::LoweringPlan &plan,
                     OpBuilder &builder,
                     Location loc,
                     Region *srcRegion = nullptr,
                     Value numThreads = nullptr,
                     Value ifClause = nullptr) {
  MLIRContext *ctx = builder.getContext();

  SmallVector<NamedAttribute> propPairs;
  for (auto &kv : plan.properties)
    propPairs.push_back(NamedAttribute(
      StringAttr::get(ctx, kv.first),
      dslValueToAttr(kv.second, ctx)));
  auto propDict = DictionaryAttr::get(ctx, propPairs);

  auto toArrayAttr = [&](const std::vector<dsl::PlanAction> &actions) {
    SmallVector<Attribute> attrs;
    for (auto &a : actions)
      attrs.push_back(planActionToAttr(a, ctx));
    return ArrayAttr::get(ctx, attrs);
  };

  // Clause operands ride a single variadic list; clause_names records which
  // clause each entry carries so the outlining pass can look them up by name
  // (a parallel can have both num_threads and if_clause at once).
  SmallVector<Value> clauseOperands;
  SmallVector<Attribute> clauseNames;
  if (numThreads) {
    clauseOperands.push_back(numThreads);
    clauseNames.push_back(builder.getStringAttr("num_threads"));
  }
  if (ifClause) {
    clauseOperands.push_back(ifClause);
    clauseNames.push_back(builder.getStringAttr("if_clause"));
  }

  auto constructOp = ConstructOp::create(builder, loc,
    clauseOperands,
    builder.getStringAttr(plan.runtime),
    builder.getStringAttr(plan.construct),
    propDict,
    toArrayAttr(plan.pre),
    toArrayAttr(plan.invoke),
    toArrayAttr(plan.post),
    clauseNames.empty() ? ArrayAttr() : builder.getArrayAttr(clauseNames));

  // Move the source region (e.g. omp.parallel body) into the construct op.
  if (srcRegion && !srcRegion->empty())
    constructOp.getBody().takeBody(*srcRegion);
}

// ===========================================================================
// Context extraction from omp.* ops
// ===========================================================================

static llvm::StringMap<dsl::Value>
extractParallelContext(omp::ParallelOp op) {
  llvm::StringMap<dsl::Value> ctx;

  ctx["body"]     = dsl::makeStr("outlined_parallel");
  ctx["captures"] = dsl::makeList({});

  // if_clause is a sentinel like on task: non-null selects `when
  // has(if_clause)` branches; the SSA value rides as a ConstructOp clause
  // operand (named "if_clause") and is resolved at the call site.
  if (op.getIfExpr())
    ctx["if_clause"] = dsl::makeStr("if_clause");
  else
    ctx["if_clause"] = dsl::makeNull();

  // num_threads is carried as a real SSA operand on the ConstructOp; in the
  // DSL the identifier is just a sentinel (resolved to the operand in
  // outlineConstruct).  Setting it to a non-null StrVal here is what makes
  // `when has(num_threads)` evaluate to true.
  if (op.getNumThreadsDimsCount() > 0)
    ctx["num_threads"] = dsl::makeStr("num_threads");
  else
    ctx["num_threads"] = dsl::makeNull();

  // proc_bind
  ctx["proc_bind"] = dsl::makeNull();
  if (op.getProcBindKind()) {
    auto pb = omp::stringifyClauseProcBindKind(*op.getProcBindKind());
    ctx["proc_bind"] = dsl::makeStr(pb.str());
  }

  // iomp runtime identifiers
  ctx["ident"]      = dsl::makeStr("%ident");
  ctx["global_tid"] = dsl::makeStr("%gtid");

  // Removed: ptr_tid/ptr_btid were only decorative args of the old microtask
  // outline_signature, which no longer exists in rules.dsl, so nothing
  // references them anymore.
  // ctx["ptr_tid"]    = dsl::makeStr("ptr_tid");
  // ctx["ptr_btid"]   = dsl::makeStr("ptr_btid");

  // Not seeded: capture_strategy values (by_pointer/packed) are bare-identifier
  // enum tokens read via evalExprOrBare, which falls back to the token's own
  // name when it isn't in this context — so seeding them would be redundant.
  // ctx["by_pointer"] = dsl::makeStr("by_pointer");
  // ctx["packed"]     = dsl::makeStr("packed");

  // Passed as a real argument by the libgomp/pmsis parallel
  // invoke (GOMP_parallel / ext_pi_cl_team_fork).
  ctx["env_ptr"]    = dsl::makeStr("env_ptr");

  return ctx;
}

// NOTE: wsloops are always nested inside a parallel (valid OpenMP never emits a
// top-level worksharing loop), so they are moved into the parallel's ConstructOp
// region here and lowered later by OmpOutliningPass, which builds the full
// wsloop context (last/stride/chunk) itself.  There is deliberately no
// extractWsloopContext / wsloop handling in this pass.

static llvm::StringMap<dsl::Value>
extractBarrierContext(omp::BarrierOp /*op*/) {
  llvm::StringMap<dsl::Value> ctx;
  ctx["ident"]      = dsl::makeStr("%ident");
  ctx["global_tid"] = dsl::makeStr("%gtid");
  return ctx;
}

// taskwait is a leaf construct like barrier: no body, no captures, just a
// call using ident + gtid (depend/nowait clauses ignored in v1).
static llvm::StringMap<dsl::Value>
extractTaskwaitContext(omp::TaskwaitOp /*op*/) {
  llvm::StringMap<dsl::Value> ctx;
  ctx["ident"]      = dsl::makeStr("%ident");
  ctx["global_tid"] = dsl::makeStr("%gtid");
  return ctx;
}

static llvm::StringMap<dsl::Value>
extractTaskContext(omp::TaskOp op) {
  llvm::StringMap<dsl::Value> ctx;

  // Resolved to the outlined function pointer in OmpOutliningPass.
  ctx["body"]     = dsl::makeStr("body");
  ctx["captures"] = dsl::makeList({});

  // if_clause is a sentinel: its presence (non-null) selects the
  // `when has(if_clause)` invoke branch; the actual SSA value is carried as
  // the ConstructOp if_clause operand and resolved at the call site.
  if (op.getIfExpr())
    ctx["if_clause"] = dsl::makeStr("if_clause");
  else
    ctx["if_clause"] = dsl::makeNull();

  // Not seeded: capture_strategy's bare `packed`/`shareds` token falls back to
  // its own name via evalExprOrBare, so it needs no context entry.  env_ptr, by
  // contrast, is used by the libgomp task invoke (GOMP_task(body, env_ptr, ...)).
  // ctx["packed"]    = dsl::makeStr("packed");
  ctx["env_ptr"]   = dsl::makeStr("env_ptr");
  // Capture-struct size/alignment placeholders, materialised at the call site
  // from the actual struct type (see OmpOutliningPass).
  ctx["env_size"]  = dsl::makeStr("env_size");
  ctx["env_align"] = dsl::makeStr("env_align");

  // iomp task tokens resolved at the call site in OmpOutliningPass::outlineTaskEntry.
  // gtid/task_t commented because before were present, but only decorative
  ctx["ident"]        = dsl::makeStr("%ident");
  ctx["global_tid"]   = dsl::makeStr("%gtid");
  // ctx["gtid"]         = dsl::makeStr("gtid");
  // ctx["task_t"]       = dsl::makeStr("task_t");
  // task_flags is not seeded here: it is a `let task_flags = 1;` in the task
  // construct (rules.dsl), so the evaluator resolves it to the literal and the
  // plan carries the value directly — mirroring `default_chunk` for wsloop.
  ctx["task_size"]    = dsl::makeStr("task_size");
  ctx["shareds_size"] = dsl::makeStr("shareds_size");
  // `task` is no longer pre-seeded: it is bound explicitly by
  // `let task = call "__kmpc_omp_task_alloc"(...)` in rules.dsl (Approach B).

  return ctx;
}

// ===========================================================================
// The pass
// ===========================================================================

struct OmpToOmpLowerPass
    : public PassWrapper<OmpToOmpLowerPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OmpToOmpLowerPass)

  std::string dslFile;
  std::string runtimeName;

  OmpToOmpLowerPass(std::string dsl, std::string rt)
      : dslFile(std::move(dsl)), runtimeName(std::move(rt)) {}
  OmpToOmpLowerPass(const OmpToOmpLowerPass &) = default;

  llvm::StringRef getArgument() const override { return "omp-to-omp-lower"; }
  llvm::StringRef getDescription() const override {
    return "Lower omp.* ops to omp_lower.construct using a DSL file";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<omp_lower::OmpLoweringDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Read and parse the DSL file
    auto buf = llvm::MemoryBuffer::getFile(dslFile);
    if (!buf) {
      module.emitError("cannot open DSL file '") << dslFile << "'";
      return signalPassFailure();
    }
    auto program = dsl::parse((*buf)->getBuffer());
    if (!program) {
      module.emitError("DSL parse error: ")
        << llvm::toString(program.takeError());
      return signalPassFailure();
    }

    dsl::Evaluator evaluator(*program);

    // Collect ops before modifying the IR
    SmallVector<omp::ParallelOp> parallels;
    SmallVector<omp::BarrierOp>  barriers;
    SmallVector<omp::TaskwaitOp> taskwaits;
    SmallVector<omp::TaskOp>     tasks;
    module.walk([&](omp::ParallelOp op) { parallels.push_back(op); });
    // Collect every task, including those nested in a parallel (the common
    // `parallel { ... task ... }` pattern) or in another task.  Parallels are
    // processed first, moving their body — with any nested task ops — into a
    // ConstructOp region; the task op pointers stay valid and are converted
    // afterwards into nested ConstructOps.  Pre-order walk guarantees an outer
    // task is processed before a task nested inside it.
    module.walk([&](omp::TaskOp op) { tasks.push_back(op); });
    // Collect every barrier, nested in a parallel or not.  Those inside a
    // parallel ride into its body region and end up in the outlined function,
    // where the outlining pass binds the microtask's gtid to them; either way
    // it is PlanLoweringPass that turns them into calls.  Wsloops are the
    // exception: they are lowered entirely by the outlining pass (the loop is
    // codegen, not a plan), so they are not collected here.
    module.walk([&](omp::BarrierOp op) { barriers.push_back(op); });
    // taskwait mirrors barrier.
    module.walk([&](omp::TaskwaitOp op) {
      // v1 lowers taskwait as an unconditional, full wait: the depend and nowait
      // clauses are not modelled yet.  Warn rather than dropping them silently,
      // so a caller relying on those semantics isn't misled.
      if (op->getNumOperands() > 0 || op->hasAttr("nowait"))
        op.emitWarning("omp-to-omp-lower: taskwait depend/nowait clauses are "
                       "ignored in v1; lowering as a full, unconditional wait");
      taskwaits.push_back(op);
    });

    // Build plan and replace each op with an omp_lower.construct
    auto process = [&](auto op,
                       llvm::StringRef construct,
                       llvm::StringMap<dsl::Value> ctx,
                       Region *region = nullptr,
                       Value numThreads = nullptr,
                       Value ifClause = nullptr) -> bool {
      auto plan = evaluator.buildPlan(runtimeName, construct, ctx);
      if (!plan) {
        op.emitError("DSL evaluation failed: ")
          << llvm::toString(plan.takeError());
        signalPassFailure();
        return false;
      }
      OpBuilder builder(op);
      emitConstructOp(*plan, builder, op.getLoc(), region, numThreads, ifClause);
      op.erase();
      return true;
    };

    // Inject firstprivate source vars as explicit uses inside a construct
    // region so collectCaptures finds them (shared with parallel handling).
    auto injectFirstprivateUses = [&](auto op) {
      Region &region = op.getRegion();
      if (region.empty()) return;
      Block &entryBlock = region.front();
      OpBuilder injector(&entryBlock, entryBlock.begin());
      auto privateSyms = op.getPrivateSyms();
      for (auto [idx, privateVar] : llvm::enumerate(op.getPrivateVars())) {
        bool isFirstprivate = false;
        if (privateSyms) {
          auto symRef = llvm::cast<SymbolRefAttr>((*privateSyms)[idx]);
          if (auto recipe = SymbolTable::lookupNearestSymbolFrom<
                  omp::PrivateClauseOp>(op, symRef))
            isFirstprivate = !recipe.getCopyRegion().empty();
        }
        if (!isFirstprivate) continue;
        UnrealizedConversionCastOp::create(injector, op.getLoc(),
          TypeRange{privateVar.getType()}, ValueRange{privateVar});
      }
    };

    // Process parallels first, passing their region for later outlining.
    // Wsloops and barriers nested inside a parallel are erased with it.
    for (auto op : parallels) {
      // Before moving the region, inject the privatizer source vars as
      // explicit uses inside the region so collectCaptures finds them.
      // Only inject for firstprivate vars (which need the source value
      // copied); purely private vars just need a fresh alloca per thread.
      injectFirstprivateUses(op);
      Value numThreads;
      if (op.getNumThreadsDimsCount() > 0)
        numThreads = op.getNumThreads(0);
      if (!process(op, "parallel", extractParallelContext(op),
                   &op.getRegion(), numThreads, op.getIfExpr())) return;
    }

    // Tasks behave like parallel for outlining (closure/packed body), but
    // carry an optional if_clause operand and no num_threads.  In OpenMP,
    // task data captures are firstprivate by default, so inject the same
    // firstprivate uses before moving the region.
    for (auto op : tasks) {
      if (!op->getBlock()) continue;
      // Clauses with no lowering yet: warn instead of silently dropping them
      // (if/firstprivate are wired; pure private is diagnosed downstream).
      if (op.getFinal())
        op.emitWarning("omp task `final` clause is not supported; ignored");
      if (op.getUntied())
        op.emitWarning("omp task `untied` clause is not supported; the task "
                       "stays tied");
      if (op.getMergeable())
        op.emitWarning("omp task `mergeable` clause is not supported; ignored");
      if (op.getPriority())
        op.emitWarning("omp task `priority` clause is not supported; ignored");
      if (!op.getDependVars().empty())
        op.emitWarning("omp task `depend` clause is not supported; the "
                       "dependency is ignored");
      if (!op.getInReductionVars().empty())
        op.emitWarning("omp task `in_reduction` clause is not supported; "
                       "ignored");
      if (op.getEventHandle())
        op.emitWarning("omp task `detach` clause is not supported; ignored");
      injectFirstprivateUses(op);
      Value ifClause = op.getIfExpr();
      if (!process(op, "task", extractTaskContext(op),
                   &op.getRegion(), /*numThreads=*/nullptr, ifClause)) return;
    }

    for (auto op : barriers) {
      if (!op->getBlock()) continue;
      if (!process(op, "barrier", extractBarrierContext(op))) return;
    }

    for (auto op : taskwaits) {
      if (!op->getBlock()) continue;
      if (!process(op, "taskwait", extractTaskwaitContext(op))) return;
    }
  }
};

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

std::unique_ptr<mlir::Pass>
mlir::createOmpToOmpLowerPass(std::string dslFile, std::string runtime) {
  return std::make_unique<OmpToOmpLowerPass>(
      std::move(dslFile), std::move(runtime));
}
