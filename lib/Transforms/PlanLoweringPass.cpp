// PlanLoweringPass.cpp
//
// Converts every omp_lower.construct op into concrete func.call operations
// targeting the selected OpenMP runtime (iomp, libgomp or pmsis).
//
// This is where a plan becomes calls.  Constructs with a body are consumed
// earlier by OmpOutliningPass, which has to emit their calls itself: the
// arguments depend on outlining artifacts (the outlined function, the capture
// struct) and the `if` clause needs a branch on a runtime value, which a flat
// plan cannot express.  Everything else — barrier, taskwait — reaches this pass
// and is lowered here.

#include "OmpLowering/Transforms/PlanLoweringPass.h"
#include "OmpLowering/IR/OmpLoweringOps.h"
#include "PlanEmit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::omp_lower;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Emission state for one construct.  The default ident and the global thread id
// are materialised on first reference and then reused, so a plan naming them
// several times still emits one global and one __kmpc_global_thread_num call.
//
// Resolution goes through the shared vocabulary in PlanEmit.h — the same one the
// outlining pass uses.  Before that was shared, this pass resolved every token
// it did not recognise to an undef pointer, which is why top-level barrier and
// taskwait had to be lowered in the outlining pass instead: an undef where iomp
// expects a gtid crashes the runtime.
struct ConstructEmitter {
  ModuleOp module;
  OpBuilder &builder;
  Location loc;
  MLIRContext *ctx;
  // `emit`-declared symbols, bound as they are encountered.
  llvm::StringMap<Value> bindings;
  Value identCache;
  Value gtidCache;

  Value getIdent() {
    if (!identCache)
      identCache = getOrCreateIdent(module, builder, loc, ctx, kIdentKmpc);
    return identCache;
  }

  Value getGtid() {
    if (!gtidCache) {
      auto decl = getOrInsertDeclWithReturn(module, "__kmpc_global_thread_num",
                                            {ptrTy(ctx)}, i32Ty(ctx), builder);
      gtidCache = func::CallOp::create(builder, loc, decl,
                                       ValueRange{getIdent()}).getResult(0);
    }
    return gtidCache;
  }

  Value resolveArg(Attribute arg) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(arg))
      return arith::ConstantOp::create(builder, loc, i32Ty(ctx),
          IntegerAttr::get(i32Ty(ctx), intAttr.getInt()));

    if (auto strAttr = llvm::dyn_cast<StringAttr>(arg))
      return resolveSymbolToken(
          strAttr.getValue(), builder, loc, bindings,
          [&](uint32_t flags) {
            return resolveIdentToken(flags, module, builder, loc, ctx,
                                     [&] { return getIdent(); });
          },
          [&] { return getGtid(); });

    // Nested ArrayAttr (e.g. a captures list) – opaque pointer placeholder.
    return LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
  }

  // Lower one plan block (pre / invoke / post).
  void lowerBlock(ArrayAttr block) {
    if (!block) return;

    for (Attribute actionAttr : block) {
      if (auto emitAttr = llvm::dyn_cast<PlanEmitAttr>(actionAttr)) {
        llvm::StringRef nm = emitAttr.getSymName().getValue();
        if (!bindings.count(nm))
          bindings[nm] = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
        continue;
      }

      if (auto callAttr = llvm::dyn_cast<PlanCallAttr>(actionAttr)) {
        ArrayAttr argAttrs = callAttr.getArgs();

        SmallVector<Value> args;
        SmallVector<Type>  argTypes;
        args.reserve(argAttrs.size());
        argTypes.reserve(argAttrs.size());
        for (Attribute a : argAttrs) {
          Value v = resolveArg(a);
          args.push_back(v);
          argTypes.push_back(v.getType());
        }

        func::FuncOp decl = getOrInsertDecl(
            module, callAttr.getCallee().getValue(), argTypes, builder);
        func::CallOp::create(builder, loc, decl, args);
        continue;
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Rewrite pattern
// ---------------------------------------------------------------------------

struct ConstructOpLowering : public OpConversionPattern<ConstructOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ConstructOp op,
                                OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op.emitOpError("expected parent ModuleOp");

    OpBuilder builder(rewriter);
    builder.setInsertionPoint(op);

    ConstructEmitter emitter{module, builder, op.getLoc(), op.getContext()};
    emitter.lowerBlock(op.getPre());
    emitter.lowerBlock(op.getInvoke());
    emitter.lowerBlock(op.getPost());

    rewriter.eraseOp(op);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct PlanLoweringPass
    : public PassWrapper<PlanLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanLoweringPass)

  llvm::StringRef getArgument()    const override { return "omp-lower-plan"; }
  llvm::StringRef getDescription() const override {
    return "Lower omp_lower.construct ops to runtime func.call ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect,
                    LLVM::LLVMDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addIllegalDialect<OmpLoweringDialect>();
    target.addLegalDialect<func::FuncDialect,
                           LLVM::LLVMDialect,
                           arith::ArithDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConstructOpLowering>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> mlir::createPlanLoweringPass() {
  return std::make_unique<PlanLoweringPass>();
}

void mlir::registerPlanLoweringPass() {
  PassRegistration<PlanLoweringPass>();
}
