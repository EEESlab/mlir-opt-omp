// Converts omp.parallel / task / barrier / taskwait into omp_lower.construct
// ops by evaluating the DSL.  Wsloops stay in place, nested inside their
// parallel, and are lowered later by OmpOutliningPass.

#include "OmpLowering/Transforms/OmpToOmpLowerPass.h"
#include "OmpLowering/IR/OmpLoweringOps.h"
#include "OmpLowering/DSL/DSLParser.h"
#include "OmpLowering/DSL/DSLEvaluator.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

// --- Privatizer recipe shapes ---
// An omp.private recipe says how to build one thread's copy of a variable.
// This lowering only allocates the storage, so the two predicates below ask
// whether a recipe wants anything beyond that (init / copy / dealloc regions).

// True when init hands the slot straight back untouched: either no init region
// (a hand-written recipe) or one whose whole body is omp.yield(%alloc), which is
// what ClangIR emits for a scalar private(j).  Anything else is asking for work
// this lowering does not do.
bool hasTrivialInit(omp::PrivateClauseOp recipe) {
  Region &init = recipe.getInitRegion();
  if (init.empty()) return true;          // not written at all
  if (!init.hasOneBlock()) return false;  // branches mean it computes something
  Block &body = init.front();
  // The verifier gives this region two args: the original variable (the mold)
  // and the fresh slot.
  if (body.getNumArguments() != 2) return false;
  auto yield = llvm::dyn_cast<omp::YieldOp>(body.getTerminator());
  if (!yield) return false;
  if (&body.front() != yield.getOperation()) return false;
  return yield.getResults().size() == 1 &&
         yield.getResults()[0] == body.getArgument(1);
}

bool hasTrivialDealloc(omp::PrivateClauseOp recipe) {
  Region &dealloc = recipe.getDeallocRegion();
  if (dealloc.empty()) return true;
  if (!dealloc.hasOneBlock()) return false;
  Block &body = dealloc.front();
  auto yield = llvm::dyn_cast<omp::YieldOp>(body.getTerminator());
  if (!yield) return false;
  return &body.front() == yield.getOperation() && yield.getResults().empty();
}

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
    },
    [&](const dsl::PlanBranch &b) -> Attribute {
      auto arm = [&](const std::vector<std::shared_ptr<dsl::PlanActionBox>> &as) {
        SmallVector<Attribute> attrs;
        for (auto &a : as)
          attrs.push_back(planActionToAttr(a->action, ctx));
        return ArrayAttr::get(ctx, attrs);
      };
      return PlanBranchAttr::get(
        ctx,
        dslValueToAttr(b.cond, ctx),
        arm(b.ifTrue),
        arm(b.ifFalse));
    }
  ), action);
}

void emitConstructOp(const dsl::LoweringPlan &plan,
                     OpBuilder &builder,
                     Location loc,
                     Region *srcRegion = nullptr,
                     Value numThreads = nullptr,
                     Value ifClause = nullptr,
                     StringAttr procBind = nullptr) {
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

// Clause operands ride one variadic list; clause_names records which clause
// each entry carries (a parallel can have num_threads and if_clause at once).
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
    clauseNames.empty() ? ArrayAttr() : builder.getArrayAttr(clauseNames),
    ArrayAttr(),
    procBind);

  if (srcRegion && !srcRegion->empty())
    constructOp.getBody().takeBody(*srcRegion);
}

static llvm::StringMap<dsl::Value>
extractParallelContext(omp::ParallelOp op) {
  llvm::StringMap<dsl::Value> ctx;

  ctx["body"]     = dsl::makeStr("outlined_parallel");
  // Symbolic, not a list: captures are unknown until the outlining pass has
  // collected them, so this stays a token resolved at the call site.
  ctx["captures"] = dsl::makeStr("%captures");

  // A sentinel: non-null selects the has(if_clause) branches; the SSA value
  // rides as a ConstructOp clause operand.
  if (op.getIfExpr())
    ctx["if_clause"] = dsl::makeStr("if_clause");
  else
    ctx["if_clause"] = dsl::makeNull();

  // A sentinel too: a non-null StrVal is what makes has(num_threads) true.
  if (op.getNumThreadsDimsCount() > 0)
    ctx["num_threads"] = dsl::makeStr("num_threads");
  else
    ctx["num_threads"] = dsl::makeNull();

  // A sentinel standing for the affinity constant, which the outlining pass
  // materialises from the kind on the ConstructOp.  Seeding the kind's own
  // spelling would put "close" in the plan as a call argument, and an unknown
  // string token resolves to an undef pointer where the ABI wants an i32.
  if (op.getProcBindKind())
    ctx["proc_bind"] = dsl::makeStr("proc_bind");
  else
    ctx["proc_bind"] = dsl::makeNull();

  // The same constant as GOMP takes it: a flags word always passed, 0 for no
  // policy.  Its own token, because proc_bind above is the *clause* and null
  // when absent so has(proc_bind) can gate iomp's push call.
  ctx["proc_bind_flags"] = dsl::makeStr("proc_bind_flags");

  ctx["ident"]      = dsl::makeStr("%ident");
  ctx["global_tid"] = dsl::makeStr("%gtid");


  ctx["env_ptr"]    = dsl::makeStr("env_ptr");

  // Slots the serialized side of if hands to the microtask, whose ABI takes
  // gtid and btid by pointer.  The outlining pass makes the allocas.
  ctx["gtid_addr"]  = dsl::makeStr("gtid_addr");
  ctx["btid_addr"]  = dsl::makeStr("btid_addr");

  return ctx;
}

// Wsloops are always nested inside a parallel, so they ride into its
// ConstructOp region and OmpOutliningPass lowers them, building the full
// context itself.  Hence no wsloop handling in this pass.

static llvm::StringMap<dsl::Value>
extractBarrierContext(omp::BarrierOp /*op*/) {
  llvm::StringMap<dsl::Value> ctx;
  ctx["ident"]      = dsl::makeStr("%ident");
  ctx["global_tid"] = dsl::makeStr("%gtid");
  return ctx;
}

// taskwait is a leaf like barrier (depend/nowait ignored in v1).
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
  // Symbolic, not a list — see extractParallelContext.
  ctx["captures"] = dsl::makeStr("%captures");

  if (op.getIfExpr())
    ctx["if_clause"] = dsl::makeStr("if_clause");
  else
    ctx["if_clause"] = dsl::makeNull();

  ctx["env_ptr"]   = dsl::makeStr("env_ptr");
  // Placeholders materialised at the call site from the actual struct type.
  ctx["env_size"]  = dsl::makeStr("env_size");
  ctx["env_align"] = dsl::makeStr("env_align");

  // Resolved at the call site in OmpOutliningPass::outlineTaskEntry.
  ctx["ident"]        = dsl::makeStr("%ident");
  ctx["global_tid"]   = dsl::makeStr("%gtid");
  // task_flags is not seeded: a `let` in the task construct resolves it.
  ctx["task_size"]    = dsl::makeStr("task_size");
  ctx["shareds_size"] = dsl::makeStr("shareds_size");

  return ctx;
}

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
    registry.insert<omp_lower::OmpLoweringDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

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

    SmallVector<omp::ParallelOp> parallels;
    SmallVector<omp::BarrierOp>  barriers;
    SmallVector<omp::TaskwaitOp> taskwaits;
    SmallVector<omp::TaskOp>     tasks;
    module.walk([&](omp::ParallelOp op) { parallels.push_back(op); });
    // Every task, including those nested in a parallel or another task.
    // Parallels are processed first, moving their body — with any nested task
    // ops — into a ConstructOp region; the pointers stay valid.  The pre-order
    // walk guarantees an outer task is processed before one nested in it.
    module.walk([&](omp::TaskOp op) { tasks.push_back(op); });
    // Every barrier, nested in a parallel or not.  Those inside one ride into
    // its body region and end up in the outlined function; either way it is
    // PlanLoweringPass that turns them into calls.  Wsloops are the exception —
    // the outlining pass lowers them entirely, since a loop is codegen.
    module.walk([&](omp::BarrierOp op) { barriers.push_back(op); });
    module.walk([&](omp::TaskwaitOp op) {
      // v1 lowers taskwait as an unconditional full wait; warn rather than
      // drop the clauses silently.
      if (op->getNumOperands() > 0 || op->hasAttr("nowait"))
        op.emitWarning("omp-to-omp-lower: taskwait depend/nowait clauses are "
                       "ignored in v1; lowering as a full, unconditional wait");
      taskwaits.push_back(op);
    });

    auto process = [&](auto op,
                       llvm::StringRef construct,
                       llvm::StringMap<dsl::Value> ctx,
                       Region *region = nullptr,
                       Value numThreads = nullptr,
                       Value ifClause = nullptr,
                       StringAttr procBind = nullptr) -> bool {
      auto plan = evaluator.buildPlan(runtimeName, construct, ctx);
      if (!plan) {
        op.emitError("DSL evaluation failed: ")
          << llvm::toString(plan.takeError());
        signalPassFailure();
        return false;
      }
      OpBuilder builder(op);
      emitConstructOp(*plan, builder, op.getLoc(), region, numThreads, ifClause,
                      procBind);
      op.erase();
      return true;
    };

    // Wire the privatizer block args the construct arrived with; telling the
    // two kinds apart is the whole job.
    //
    // A firstprivate reads the original, so it must reach the outlined
    // function: inject a use inside the region for collectCaptures to find and
    // leave the block arg for OmpOutliningPass to pair with that capture.
    //
    // A pure private reads nothing, so its slot is allocated here and the block
    // arg goes with the clause operand that fed it.  Only firstprivate args
    // then reach the ConstructOp, which is what makes the outlining pass's
    // positional pairing (arg i <-> capture i) correct.
    auto wirePrivatizers = [&](auto op) {
      Region &region = op.getRegion();
      if (region.empty() || op.getPrivateVars().empty()) return true;
      Block &entryBlock = region.front();
      OpBuilder injector(&entryBlock, entryBlock.begin());
      auto privateSyms = op.getPrivateSyms();
      auto blockArgs =
          llvm::cast<omp::BlockArgOpenMPOpInterface>(op.getOperation())
              .getPrivateBlockArgs();

      auto fail = [&](const char *msg) {
        op.emitError("omp-to-omp-lower: ") << msg;
        signalPassFailure();
      };

      Value one;   // the alloca count, materialised on first use
      // Per pure private: index in private_vars and the block arg it feeds,
      // both read before any erase invalidates them.
      SmallVector<std::pair<unsigned, unsigned>> dropped;
      bool ok = true;
      for (auto [idx, privateVar] : llvm::enumerate(op.getPrivateVars())) {
        omp::PrivateClauseOp recipe;
        if (privateSyms)
          recipe = SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
              op, llvm::cast<SymbolRefAttr>((*privateSyms)[idx]));
        if (!recipe) {
          fail("privatizer recipe not found, so `private` cannot be told from "
               "`firstprivate`");
          ok = false;
          continue;
        }
        // Declared firstprivate, or carrying the copy region that does the
        // copy-in: either way the source is read and must be captured.
        if (recipe.getDataSharingType() ==
                omp::DataSharingClauseType::FirstPrivate ||
            !recipe.getCopyRegion().empty()) {
          UnrealizedConversionCastOp::create(injector, op.getLoc(),
            TypeRange{privateVar.getType()}, ValueRange{privateVar});
          continue;
        }

        // An init or dealloc that *does* something means the slot takes more
        // than storage, so a bare alloca would be the wrong lowering.
        if (!hasTrivialInit(recipe) || !hasTrivialDealloc(recipe)) {
          fail("`private` clause whose recipe has a non-trivial init or "
               "dealloc region is not supported");
          ok = false;
          continue;
        }
        BlockArgument arg = blockArgs[idx];
        Type slotTy = recipe.getType();
        if (!llvm::isa<LLVM::LLVMPointerType>(arg.getType()) ||
            !LLVM::isCompatibleType(slotTy)) {
          fail("`private` clause of an unsupported type: the region argument "
               "must be a pointer to LLVM-compatible storage");
          ok = false;
          continue;
        }
        if (!one) {
          auto i64Ty = injector.getI64Type();
          one = LLVM::ConstantOp::create(injector, op.getLoc(), i64Ty,
                                         injector.getI64IntegerAttr(1));
        }
        // One slot per execution of the outlined body, i.e. one per thread.
        Value slot = LLVM::AllocaOp::create(injector, op.getLoc(),
                                            arg.getType(), slotTy, one);
        arg.replaceAllUsesWith(slot);
        dropped.push_back({(unsigned)idx, arg.getArgNumber()});
      }

      // Back to front so earlier indices stay valid.  The op is erased right
      // after but stays verifiable until then: arg count and private_vars agree.
      if (!dropped.empty()) {
        llvm::SmallDenseSet<unsigned> droppedVars;
        for (auto [varIdx, argNo] : dropped) droppedVars.insert(varIdx);
        for (auto [varIdx, argNo] : llvm::reverse(dropped)) {
          entryBlock.eraseArgument(argNo);
          op.getPrivateVarsMutable().erase(varIdx);
        }
        if (privateSyms) {
          SmallVector<Attribute> kept;
          for (auto [i, sym] : llvm::enumerate(*privateSyms))
            if (!droppedVars.contains((unsigned)i)) kept.push_back(sym);
          if (kept.empty())
            op.removePrivateSymsAttr();
          else
            op.setPrivateSymsAttr(ArrayAttr::get(op.getContext(), kept));
        }
      }
      return ok;
    };

    // Parallels first, passing their region for later outlining.
    for (auto op : parallels) {
      // Capture the firstprivate sources and give each pure private its own slot.
      if (!wirePrivatizers(op)) return;
      Value numThreads;
      if (op.getNumThreadsDimsCount() > 0)
        numThreads = op.getNumThreads(0);
      // The kind's spelling, not its ordinal: the MLIR enum numbers its cases
      // differently from every runtime's.
      StringAttr procBind;
      if (auto kind = op.getProcBindKind())
        procBind = StringAttr::get(&getContext(),
                                   omp::stringifyClauseProcBindKind(*kind));
      if (!process(op, "parallel", extractParallelContext(op),
                   &op.getRegion(), numThreads, op.getIfExpr(), procBind))
        return;
    }

    // Tasks outline like parallel, but with an optional if_clause and no
    // num_threads.
    for (auto op : tasks) {
      if (!op->getBlock()) continue;
      // Clauses with no lowering yet: warn rather than drop them silently.
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
      if (op.getEventHandle())
        op.emitWarning("omp task `detach` clause is not supported; ignored");
      if (!wirePrivatizers(op)) return;
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


std::unique_ptr<mlir::Pass>
mlir::createOmpToOmpLowerPass(std::string dslFile, std::string runtime) {
  return std::make_unique<OmpToOmpLowerPass>(
      std::move(dslFile), std::move(runtime));
}
