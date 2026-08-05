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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

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
  // Tokens that stand for a whole list of values rather than one — `captures`
  // being the case that matters.  A list token used as a call argument splices
  // into that many arguments, and `argc(<list>)` becomes its length.  Neither
  // can be settled while the rules are evaluated, because the outlining pass
  // has not collected the captures yet.
  llvm::StringMap<SmallVector<Value>> listBindings;
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

    if (auto strAttr = llvm::dyn_cast<StringAttr>(arg)) {
      // `argc(<list>)` arrives as "%argc:<name>": the length of a list binding,
      // known only now.
      llvm::StringRef s = strAttr.getValue();
      if (s.consume_front("%argc:")) {
        auto it = listBindings.find(("%" + s).str());
        int64_t n = it == listBindings.end() ? 0 : (int64_t)it->second.size();
        return arith::ConstantOp::create(builder, loc, i32Ty(ctx),
            IntegerAttr::get(i32Ty(ctx), n));
      }
      return resolveSymbolToken(
          s, builder, loc, bindings,
          [&](uint32_t flags) {
            return resolveIdentToken(flags, module, builder, loc, ctx,
                                     [&] { return getIdent(); });
          },
          [&] { return getGtid(); });
    }

    // Nested ArrayAttr (e.g. a captures list) – opaque pointer placeholder.
    return LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
  }

  void lowerCall(PlanCallAttr callAttr) {
    ArrayAttr argAttrs = callAttr.getArgs();
    SmallVector<Value> args;
    SmallVector<Type>  argTypes;
    // Where the first spliced list started, i.e. how many arguments are fixed.
    // A call that splices a list is variadic by construction: the same callee
    // is reached from parallel regions with different capture counts, so it
    // cannot be declared with one fixed signature.
    std::optional<size_t> fixedArgs;
    args.reserve(argAttrs.size());
    argTypes.reserve(argAttrs.size());
    for (Attribute a : argAttrs) {
      // A list token contributes as many arguments as it is bound to, not one.
      if (auto s = llvm::dyn_cast<StringAttr>(a)) {
        auto it = listBindings.find(s.getValue());
        if (it != listBindings.end()) {
          if (!fixedArgs) fixedArgs = args.size();
          for (Value v : it->second) {
            args.push_back(v);
            argTypes.push_back(v.getType());
          }
          continue;
        }
      }
      Value v = resolveArg(a);
      args.push_back(v);
      argTypes.push_back(v.getType());
    }

    llvm::StringRef callee = callAttr.getCallee().getValue();

    // A callee that names a binding is a function *value*, not a symbol: the
    // outlined function's real name is only known to the outlining pass, which
    // hands its address over as `body`.  Call through the pointer.
    if (auto bound = bindings.find(callee); bound != bindings.end()) {
      auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                              argTypes, /*isVarArg=*/false);
      // The indirect form takes the callee pointer as the first operand rather
      // than as a separate parameter.
      SmallVector<Value> operands;
      operands.push_back(bound->second);
      operands.append(args.begin(), args.end());
      LLVM::CallOp::create(builder, loc, fnTy, operands);
      return;
    }

    if (fixedArgs) {
      auto fnTy = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx),
          ArrayRef<Type>(argTypes).take_front(*fixedArgs), /*isVarArg=*/true);
      LLVM::LLVMFuncOp decl =
          module.lookupSymbol<LLVM::LLVMFuncOp>(callee);
      if (!decl) {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(module.getBody());
        decl = LLVM::LLVMFuncOp::create(builder, UnknownLoc::get(ctx), callee,
                                        fnTy, LLVM::Linkage::External);
      }
      LLVM::CallOp::create(builder, loc, decl, args);
      return;
    }

    func::FuncOp decl = getOrInsertDecl(module, callee, argTypes, builder);
    func::CallOp::create(builder, loc, decl, args);
  }

  // `branch <cond> { true => ... false => ... }`: the one plan action that is
  // not a straight line.  Splits the current block in two and fills a block per
  // arm, leaving the builder at the top of the continuation so whatever follows
  // in the plan lands after the join.
  // Every string token appearing as a call argument, or as a nested branch's
  // condition, anywhere under this block.
  static void forEachToken(ArrayAttr block,
                           llvm::function_ref<void(llvm::StringRef)> fn) {
    if (!block) return;
    for (Attribute a : block) {
      if (auto c = llvm::dyn_cast<PlanCallAttr>(a)) {
        for (Attribute arg : c.getArgs())
          if (auto s = llvm::dyn_cast<StringAttr>(arg)) fn(s.getValue());
      } else if (auto b = llvm::dyn_cast<PlanBranchAttr>(a)) {
        if (auto s = llvm::dyn_cast<StringAttr>(b.getCond())) fn(s.getValue());
        forEachToken(b.getIfTrue(), fn);
        forEachToken(b.getIfFalse(), fn);
      }
    }
  }

  void lowerBranch(PlanBranchAttr br) {
    Value cond = clauseToI1(builder, loc, resolveArg(br.getCond()));

    // The default ident and the gtid are materialised on first use and reused.
    // Left to the arms, whichever referenced one first would define it inside
    // its own block, where the other arm cannot see it.  Force them out here,
    // before the split, if either arm asks for them — and only then, so a
    // branch that needs neither (libgomp's, pmsis's) still emits neither.
    bool needsIdent = false, needsGtid = false;
    auto scan = [&](llvm::StringRef s) {
      uint32_t flags;
      if (parseIdentRef(s, flags)) needsIdent = true;
      if (s == "%gtid") needsGtid = true;
    };
    forEachToken(br.getIfTrue(), scan);
    forEachToken(br.getIfFalse(), scan);
    if (needsGtid) getGtid();        // seeds off the ident itself
    else if (needsIdent) getIdent();

    Block *curBlock = builder.getInsertionBlock();
    Block *contBlock = curBlock->splitBlock(builder.getInsertionPoint());
    Block *trueBlock  = builder.createBlock(contBlock);
    Block *falseBlock = builder.createBlock(contBlock);

    builder.setInsertionPointToEnd(curBlock);
    LLVM::CondBrOp::create(builder, loc, cond, trueBlock, falseBlock);

    for (auto [blk, arm] : {std::pair{trueBlock,  br.getIfTrue()},
                            std::pair{falseBlock, br.getIfFalse()}}) {
      builder.setInsertionPointToEnd(blk);
      lowerActions(arm);
      LLVM::BrOp::create(builder, loc, contBlock);
    }

    builder.setInsertionPointToStart(contBlock);
  }

  void lowerActions(ArrayAttr block) {
    if (!block) return;

    for (Attribute actionAttr : block) {
      if (auto emitAttr = llvm::dyn_cast<PlanEmitAttr>(actionAttr)) {
        llvm::StringRef nm = emitAttr.getSymName().getValue();
        if (!bindings.count(nm))
          bindings[nm] = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
        continue;
      }
      if (auto callAttr = llvm::dyn_cast<PlanCallAttr>(actionAttr)) {
        lowerCall(callAttr);
        continue;
      }
      if (auto branchAttr = llvm::dyn_cast<PlanBranchAttr>(actionAttr)) {
        lowerBranch(branchAttr);
        continue;
      }
    }
  }

  // Lower one plan block (pre / invoke / post).
  void lowerBlock(ArrayAttr block) { lowerActions(block); }
};

// ---------------------------------------------------------------------------
// Lowering one construct
// ---------------------------------------------------------------------------

// A plain walk rather than a dialect conversion: `branch` splits the block it
// is emitted into, and the conversion framework tracks IR changes in a way that
// direct block surgery does not fit.  The outlining pass emits its own control
// flow the same way.
static LogicalResult lowerConstruct(ConstructOp op) {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op.emitOpError("expected parent ModuleOp");

    OpBuilder builder(op);
    builder.setInsertionPoint(op);

    ConstructEmitter emitter{module, builder, op.getLoc(), op.getContext()};

    // Seed the bindings from the construct's operands.  The 1:1 names array
    // says what each one is: a clause value ("num_threads", "if_clause") or a
    // value only the earlier passes could produce — notably "%gtid", which the
    // outlining pass binds to the microtask's thread id for leaf constructs
    // that ended up inside an outlined function.  Bindings are consulted first,
    // so a bound %gtid wins over materialising a fresh __kmpc_global_thread_num.
    // Names declared as lists get an entry even when no operand carries them,
    // so an empty list stays a list: `captures` on a parallel that captures
    // nothing must splice into zero arguments, not fall through to undef.
    llvm::StringSet<> isList;
    if (auto lists = op.getListNames())
      for (Attribute n : *lists) {
        auto name = llvm::cast<StringAttr>(n).getValue();
        isList.insert(name);
        emitter.listBindings[name];
      }

    if (auto names = op.getClauseNames()) {
      auto operands = op.getClauseOperands();
      for (auto [i, n] : llvm::enumerate(*names)) {
        if (i >= operands.size()) break;
        auto name = llvm::cast<StringAttr>(n).getValue();
        if (isList.contains(name))
          emitter.listBindings[name].push_back(operands[i]);
        else
          emitter.bindings[name] = operands[i];
      }
    }

    emitter.lowerBlock(op.getPre());
    emitter.lowerBlock(op.getInvoke());
    emitter.lowerBlock(op.getPost());

    op.erase();
    return success();
}

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

    // Collect first: lowering a construct rewrites the block it lives in.
    SmallVector<ConstructOp> constructs;
    module.walk([&](ConstructOp op) { constructs.push_back(op); });

    for (auto op : constructs)
      if (failed(lowerConstruct(op)))
        return signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> mlir::createPlanLoweringPass() {
  return std::make_unique<PlanLoweringPass>();
}

void mlir::registerPlanLoweringPass() {
  PassRegistration<PlanLoweringPass>();
}
