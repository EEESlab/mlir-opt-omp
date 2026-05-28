// OmpToOmpLowerPass.cpp
//
// Converts omp.parallel / omp.wsloop / omp.barrier ops to
// omp_lower.construct ops by evaluating the user-provided DSL file.

#include "OmpToOmpLowerPass.h"
#include "OmpLoweringOps.h"
#include "DSLParser.h"
#include "DSLEvaluator.h"

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
        ArrayAttr::get(ctx, argAttrs));
    }
  ), action);
}

void emitConstructOp(const dsl::LoweringPlan &plan,
                     OpBuilder &builder,
                     Location loc,
                     Region *srcRegion = nullptr) {
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

  auto constructOp = ConstructOp::create(builder, loc,
    builder.getStringAttr(plan.runtime),
    builder.getStringAttr(plan.construct),
    propDict,
    toArrayAttr(plan.pre),
    toArrayAttr(plan.invoke),
    toArrayAttr(plan.post));

  // Move the source region (e.g. omp.parallel body) into the construct op.
  if (srcRegion && !srcRegion->empty())
    constructOp.getBody().takeBody(*srcRegion);
}

// ===========================================================================
// Context extraction from omp.* ops
// ===========================================================================

static dsl::Value ssaToStrVal(mlir::Value v) {
  if (!v) return dsl::makeNull();
  std::string s;
  llvm::raw_string_ostream os(s);
  v.print(os);
  return dsl::makeStr(s);
}

static llvm::StringMap<dsl::Value>
extractParallelContext(omp::ParallelOp op) {
  llvm::StringMap<dsl::Value> ctx;

  ctx["body"]     = dsl::makeStr("outlined_parallel");
  ctx["captures"] = dsl::makeList({});

  // if_clause
  if (auto ifVar = op.getIfExpr())
    ctx["if_clause"] = ssaToStrVal(ifVar);
  else
    ctx["if_clause"] = dsl::makeNull();

  // num_threads is variadic in newer MLIR – guard with count check
  if (op.getNumThreadsDimsCount() > 0)
    ctx["num_threads"] = ssaToStrVal(op.getNumThreads(0));
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
  ctx["global_tid"] = dsl::makeStr("%tid");
  ctx["ptr_tid"]    = dsl::makeStr("ptr_tid");
  ctx["ptr_btid"]   = dsl::makeStr("ptr_btid");

  // capture strategy enum values used as bare identifiers in DSL
  ctx["by_pointer"] = dsl::makeStr("by_pointer");
  ctx["packed"]     = dsl::makeStr("packed");
  ctx["env_ptr"]    = dsl::makeStr("env_ptr");

  return ctx;
}

static llvm::StringMap<dsl::Value>
extractWsloopContext(omp::WsloopOp op) {
  llvm::StringMap<dsl::Value> ctx;
  ctx["ident"]      = dsl::makeStr("%ident");
  ctx["global_tid"] = dsl::makeStr("%tid");
  ctx["lower"]      = dsl::makeStr("%lb");
  ctx["upper"]      = dsl::makeStr("%ub");
  ctx["step"]       = dsl::makeStr("%step");
  ctx["chunk"]      = dsl::makeInt(1);
  ctx["nowait"]     = dsl::makeBool(op.getNowait());

  // schedule
  ctx["schedule"] = dsl::makeStr("static");
  if (op.getScheduleKind()) {
    auto sk = omp::stringifyClauseScheduleKind(*op.getScheduleKind());
    ctx["schedule"] = dsl::makeStr(sk.str());
  }

  return ctx;
}

static llvm::StringMap<dsl::Value>
extractBarrierContext(omp::BarrierOp /*op*/) {
  llvm::StringMap<dsl::Value> ctx;
  ctx["ident"]      = dsl::makeStr("%ident");
  ctx["global_tid"] = dsl::makeStr("%tid");
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
    SmallVector<omp::WsloopOp>   wsloops;
    SmallVector<omp::BarrierOp>  barriers;
    module.walk([&](omp::ParallelOp op) { parallels.push_back(op); });
    // Only collect wsloops and barriers that are NOT nested inside a parallel.
    // Those inside a parallel are part of its body region and will be handled
    // by the outlining pass after the parallel body is moved.
    module.walk([&](omp::WsloopOp op) {
      if (!op->getParentOfType<omp::ParallelOp>())
        wsloops.push_back(op);
    });
    module.walk([&](omp::BarrierOp op) {
      if (!op->getParentOfType<omp::ParallelOp>())
        barriers.push_back(op);
    });

    // Build plan and replace each op with an omp_lower.construct
    auto process = [&](auto op,
                       llvm::StringRef construct,
                       llvm::StringMap<dsl::Value> ctx,
                       Region *region = nullptr) -> bool {
      auto plan = evaluator.buildPlan(runtimeName, construct, ctx);
      if (!plan) {
        op.emitError("DSL evaluation failed: ")
          << llvm::toString(plan.takeError());
        signalPassFailure();
        return false;
      }
      OpBuilder builder(op);
      emitConstructOp(*plan, builder, op.getLoc(), region);
      op.erase();
      return true;
    };

    // Process parallels first, passing their region for later outlining.
    // Wsloops and barriers nested inside a parallel are erased with it.
    for (auto op : parallels) {
      // Before moving the region, inject the privatizer source vars as
      // explicit uses inside the region so collectCaptures finds them.
      // Only inject for firstprivate vars (which need the source value
      // copied); purely private vars just need a fresh alloca per thread.
      Region &parallelRegion = op.getRegion();
      if (!parallelRegion.empty()) {
        Block &entryBlock = parallelRegion.front();
        OpBuilder injector(&entryBlock, entryBlock.begin());
        auto privateSyms = op.getPrivateSyms();
        for (auto [idx, privateVar] : llvm::enumerate(op.getPrivateVars())) {
          // Look up the privatizer recipe to check if it's firstprivate.
          bool isFirstprivate = false;
          if (privateSyms) {
            auto symRef = llvm::cast<SymbolRefAttr>((*privateSyms)[idx]);
            if (auto recipe = SymbolTable::lookupNearestSymbolFrom<
                    omp::PrivateClauseOp>(op, symRef))
              isFirstprivate = !recipe.getCopyRegion().empty();
          }
          if (!isFirstprivate) continue;
          // Create a use of privateVar inside the region so collectCaptures
          // picks it up as a capture.
          UnrealizedConversionCastOp::create(injector, op.getLoc(),
            TypeRange{privateVar.getType()}, ValueRange{privateVar});
        }
      }
      if (!process(op, "parallel", extractParallelContext(op),
                   &op.getRegion())) return;
    }

    for (auto op : wsloops) {
      if (!op->getBlock()) continue;
      if (!process(op, "wsloop", extractWsloopContext(op))) return;
    }

    for (auto op : barriers) {
      if (!op->getBlock()) continue;
      if (!process(op, "barrier", extractBarrierContext(op))) return;
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
