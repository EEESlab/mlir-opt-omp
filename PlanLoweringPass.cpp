// PlanLoweringPass.cpp
//
// Converts every omp_lower.construct op into concrete func.call operations
// targeting the selected OpenMP runtime (iomp or libgomp).

#include "PlanLoweringPass.h"
#include "OmpLoweringOps.h"

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

static Type opaquePtr(MLIRContext *ctx) {
  return LLVM::LLVMPointerType::get(ctx);
}

// Resolve a plan argument attribute to an SSA Value.
static Value resolveArg(Attribute arg,
                        llvm::StringMap<Value> &valueMap,
                        OpBuilder &builder,
                        Location loc) {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(arg))
    return arith::ConstantOp::create(builder, loc,
        builder.getI32Type(),
        builder.getI32IntegerAttr(
            static_cast<int32_t>(intAttr.getInt())));

  if (auto strAttr = llvm::dyn_cast<StringAttr>(arg)) {
    llvm::StringRef nm = strAttr.getValue();
    auto it = valueMap.find(nm);
    if (it != valueMap.end())
      return it->second;
    Value ph = LLVM::UndefOp::create(builder, loc, opaquePtr(builder.getContext()));
    valueMap[nm] = ph;
    return ph;
  }

  // Nested ArrayAttr (e.g. a captures list) – opaque pointer placeholder.
  return LLVM::UndefOp::create(builder, loc, opaquePtr(builder.getContext()));
}

// Ensure a private external func declaration exists for the callee,
// using the actual argument types provided.
static func::FuncOp getOrInsertRuntimeDecl(ModuleOp module,
                                           llvm::StringRef callee,
                                           ArrayRef<Type> argTypes,
                                           OpBuilder &builder) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(callee))
    return existing;

  MLIRContext *ctx = module.getContext();
  auto fnType = builder.getFunctionType(argTypes, {});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto decl = func::FuncOp::create(builder.getUnknownLoc(), callee, fnType);
  module.getBody()->push_back(decl);
  decl.setVisibility(SymbolTable::Visibility::Private);
  decl->setAttr("llvm.linkage",
                LLVM::LinkageAttr::get(ctx, LLVM::Linkage::External));
  return decl;
}

// Lower one plan block (pre / invoke / post).
static void lowerBlock(ArrayAttr block,
                       ModuleOp module,
                       OpBuilder &builder,
                       Location loc,
                       llvm::StringMap<Value> &valueMap) {
  if (!block) return;

  for (Attribute actionAttr : block) {
    if (auto emitAttr = llvm::dyn_cast<PlanEmitAttr>(actionAttr)) {
      llvm::StringRef nm = emitAttr.getSymName().getValue();
      if (!valueMap.count(nm)) {
        Value v = LLVM::UndefOp::create(builder, loc,
                      opaquePtr(builder.getContext()));
        valueMap[nm] = v;
      }
      continue;
    }

    if (auto callAttr = llvm::dyn_cast<PlanCallAttr>(actionAttr)) {
      llvm::StringRef callee = callAttr.getCallee().getValue();
      ArrayAttr argAttrs = callAttr.getArgs();

      SmallVector<Value> args;
      args.reserve(argAttrs.size());
      for (Attribute a : argAttrs)
        args.push_back(resolveArg(a, valueMap, builder, loc));

      // Collect the actual types from the resolved SSA values.
      SmallVector<Type> argTypes;
      argTypes.reserve(args.size());
      for (auto v : args)
        argTypes.push_back(v.getType());

      func::FuncOp decl =
          getOrInsertRuntimeDecl(module, callee, argTypes, builder);
      func::CallOp::create(builder, loc, decl, args);
      continue;
    }
  }
}

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

    llvm::StringMap<Value> valueMap;
    OpBuilder builder(rewriter);
    builder.setInsertionPoint(op);
    Location loc = op.getLoc();

    lowerBlock(op.getPre(),    module, builder, loc, valueMap);
    lowerBlock(op.getInvoke(), module, builder, loc, valueMap);
    lowerBlock(op.getPost(),   module, builder, loc, valueMap);

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
