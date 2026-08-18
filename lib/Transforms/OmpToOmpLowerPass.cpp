// OmpToOmpLowerPass.cpp
//
// Converts omp.parallel / omp.task / omp.barrier / omp.taskwait ops to omp_lower.construct
// ops by evaluating the user-provided DSL file.  Wsloops are left in place
// (nested inside their parallel) and lowered later by OmpOutliningPass.

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

// ===========================================================================
// Privatizer recipe shapes
// ===========================================================================
//
// An omp.private recipe says how to build one thread's own copy of a variable.
// The storage is allocated by whoever lowers the clause, here
// wirePrivatizers, and the recipe's three optional regions then act on it:
// `init` prepares the fresh slot, `copy` fills it from the original (which is
// what makes a firstprivate), `dealloc` tears it down afterwards.
//
// This lowering only allocates.  The two predicates below ask whether a recipe
// is content with that — whether it wants anything beyond the bare storage.

// True when the init region hands the slot straight back without touching it.
// Two spellings mean that: no init region at all, which is how a recipe is
// written by hand, and a region whose whole body is `omp.yield(%alloc)`, which
// is what ClangIR emits for a scalar `private(j)`.
//
// A region that does anything else — zeroing the slot, sizing it from the
// original — is asking for work this lowering does not perform.
bool hasTrivialInit(omp::PrivateClauseOp recipe) {
  Region &init = recipe.getInitRegion();
  if (init.empty()) return true;          // not written at all
  if (!init.hasOneBlock()) return false;  // branches mean it computes something
  Block &body = init.front();
  // The verifier gives this region two arguments: the original variable (the
  // "mold", there to be read for things like an array's length) and the fresh
  // slot.
  if (body.getNumArguments() != 2) return false;
  auto yield = llvm::dyn_cast<omp::YieldOp>(body.getTerminator());
  if (!yield) return false;
  // Nothing but the terminator, or the extra ops are work done to the slot.
  if (&body.front() != yield.getOperation()) return false;
  // And what it yields is that same slot, not something it built instead.
  return yield.getResults().size() == 1 &&
         yield.getResults()[0] == body.getArgument(1);
}

// The same for dealloc: no region, or one that ends without having done
// anything, means the slot needs no tearing down.
bool hasTrivialDealloc(omp::PrivateClauseOp recipe) {
  Region &dealloc = recipe.getDeallocRegion();
  if (dealloc.empty()) return true;
  if (!dealloc.hasOneBlock()) return false;
  Block &body = dealloc.front();
  auto yield = llvm::dyn_cast<omp::YieldOp>(body.getTerminator());
  if (!yield) return false;
  return &body.front() == yield.getOperation() && yield.getResults().empty();
}

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
    clauseNames.empty() ? ArrayAttr() : builder.getArrayAttr(clauseNames),
    // list_names: the outlining pass sets it when it binds a list (captures).
    ArrayAttr(),
    // proc_bind: a compile-time enum, so it travels as an attribute rather
    // than as one of the clause operands above.
    procBind);

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
  // Symbolic, not a list: the captures are not known until the outlining
  // pass has collected them, so `captures` and `argc(captures)` stay tokens
  // and are resolved against the bindings when the call is emitted.
  ctx["captures"] = dsl::makeStr("%captures");

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

  // proc_bind is a sentinel like num_threads: non-null selects the
  // `when has(proc_bind)` push call, and the identifier in the DSL stands for
  // the affinity constant, which the outlining pass materialises from the kind
  // carried on the ConstructOp.  Seeding the kind's own spelling here instead
  // would put "close" in the plan as a call argument, and an unknown string
  // token resolves to an undef pointer where the ABI wants an i32 enum.
  if (op.getProcBindKind())
    ctx["proc_bind"] = dsl::makeStr("proc_bind");
  else
    ctx["proc_bind"] = dsl::makeNull();

  // The same affinity constant seen the way GOMP takes it: a flags word that is
  // always passed, 0 meaning "no policy asked for" — the value GCC itself emits
  // there.  It needs its own token because `proc_bind` above is the *clause*,
  // null when absent so `has(proc_bind)` can gate iomp's push call, and a null
  // resolves to a null pointer where this slot wants an i32.
  ctx["proc_bind_flags"] = dsl::makeStr("proc_bind_flags");

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

  // Slots the serialized side of `if` hands to the microtask, whose ABI takes
  // gtid and btid by pointer.  Tokens here; the outlining pass makes the
  // allocas and binds them, since only it knows the outlined signature.
  ctx["gtid_addr"]  = dsl::makeStr("gtid_addr");
  ctx["btid_addr"]  = dsl::makeStr("btid_addr");

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
  // Symbolic, not a list: the captures are not known until the outlining
  // pass has collected them, so `captures` and `argc(captures)` stay tokens
  // and are resolved against the bindings when the call is emitted.
  ctx["captures"] = dsl::makeStr("%captures");

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
    registry.insert<omp_lower::OmpLoweringDialect, LLVM::LLVMDialect>();
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

    // Wire the privatizer block args a construct arrived with.  The two kinds
    // need opposite things, and telling them apart is the whole job here.
    //
    // A firstprivate reads the original value, so it has to reach the outlined
    // function: inject a use of it inside the region for collectCaptures to
    // find, and leave the block arg for OmpOutliningPass to pair with that
    // capture.
    //
    // A pure private reads nothing — each thread just needs its own slot — so
    // the slot is allocated here, in the region about to become the outlined
    // body, and the block arg goes with the clause operand that fed it.  Only
    // firstprivate args then reach the ConstructOp, which is what makes the
    // outlining pass's positional pairing (arg i <-> capture i) correct.
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
      // Per pure private: index in private_vars, and the block arg it feeds.
      // Both are read before any erase invalidates them.
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

        // An init or dealloc region that *does* something means the slot takes
        // more than storage — a descriptor to set up, a destructor to run — so
        // a bare alloca would be the wrong lowering, not an incomplete one.
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
        // One slot per execution of the outlined body, which is one per thread.
        Value slot = LLVM::AllocaOp::create(injector, op.getLoc(),
                                            arg.getType(), slotTy, one);
        arg.replaceAllUsesWith(slot);
        dropped.push_back({(unsigned)idx, arg.getArgNumber()});
      }

      // Drop the pure privates: block arg, clause operand and private_syms
      // entry, back to front so earlier indices stay valid.  The op is erased
      // right after, but stays verifiable until then — arg count and
      // private_vars have to agree.
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

    // Process parallels first, passing their region for later outlining.
    // Wsloops and barriers nested inside a parallel are erased with it.
    for (auto op : parallels) {
      // Before moving the region: capture the firstprivate sources and give
      // each pure private its own slot inside the region (see wirePrivatizers).
      if (!wirePrivatizers(op)) return;
      Value numThreads;
      if (op.getNumThreadsDimsCount() > 0)
        numThreads = op.getNumThreads(0);
      // The kind's spelling, not its ordinal: the MLIR enum numbers its cases
      // differently from every runtime's, so the name is what survives.
      StringAttr procBind;
      if (auto kind = op.getProcBindKind())
        procBind = StringAttr::get(&getContext(),
                                   omp::stringifyClauseProcBindKind(*kind));
      if (!process(op, "parallel", extractParallelContext(op),
                   &op.getRegion(), numThreads, op.getIfExpr(), procBind))
        return;
    }

    // Tasks behave like parallel for outlining (closure/packed body), but
    // carry an optional if_clause operand and no num_threads.  In OpenMP,
    // task data captures are firstprivate by default, so inject the same
    // firstprivate uses before moving the region.
    for (auto op : tasks) {
      if (!op->getBlock()) continue;
      // Clauses with no lowering yet: warn instead of silently dropping them
      // (if, private and firstprivate are wired).
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

// ===========================================================================
// Public API
// ===========================================================================

std::unique_ptr<mlir::Pass>
mlir::createOmpToOmpLowerPass(std::string dslFile, std::string runtime) {
  return std::make_unique<OmpToOmpLowerPass>(
      std::move(dslFile), std::move(runtime));
}
