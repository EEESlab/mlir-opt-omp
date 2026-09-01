// Turns every omp_lower.construct into concrete runtime calls: parallel, task,
// barrier, taskwait.  OmpOutliningPass runs first and attaches what only it can
// produce (the outlined function, capture values, ABI sizes), deciding nothing
// about the call sequence — so a runtime is retargeted by editing rules.dsl
// alone.
//
// wsloop is the exception: it never becomes a construct, because its pre and
// post straddle a loop skeleton the outlining pass builds, and the bounds
// depend on which shape pre took.  Its callees still come from the plan.

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

namespace {

// Emission state for one construct.  The default ident and the thread id are
// materialised on first reference and reused, so a plan naming them repeatedly
// still emits one global and one gtid call.  Token resolution goes through the
// shared vocabulary in PlanEmit.h, the same one the outlining pass uses.
struct ConstructEmitter {
  ModuleOp module;
  OpBuilder &builder;
  Location loc;
  MLIRContext *ctx;
  llvm::StringMap<Value> bindings;
  // Tokens standing for a whole list rather than one value — captures being the
  // case that matters.  A list token splices into that many call arguments and
  // argc(<list>) becomes its length; neither can be settled during evaluation,
  // since the outlining pass has not collected the captures yet.
  llvm::StringMap<SmallVector<Value>> listBindings;
  // The construct's kmp_task_t layout, when it declares one (iomp task).
  LLVM::LLVMStructType kmpTaskTy;
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

  // C _Bool is i8, so a boolean literal and the if-clause value are widened
  // from the i1 the rest of the IR uses.
  Value toI8(Value v) {
    auto i8t = IntegerType::get(ctx, 8);
    if (v.getType() == i8t) return v;
    unsigned bw = v.getType().getIntOrFloatBitWidth();
    return bw < 8 ? LLVM::ZExtOp::create(builder, loc, i8t, v).getResult()
                  : LLVM::TruncOp::create(builder, loc, i8t, v).getResult();
  }

  Value resolveArg(Attribute arg) {
    // A bool literal is a C _Bool, not an i1: GOMP_task's if_clause slot.
    if (auto boolAttr = llvm::dyn_cast<BoolAttr>(arg)) {
      auto i8t = IntegerType::get(ctx, 8);
      return LLVM::ConstantOp::create(builder, loc, i8t,
          IntegerAttr::get(i8t, boolAttr.getValue() ? 1 : 0));
    }

    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(arg))
      return arith::ConstantOp::create(builder, loc, i32Ty(ctx),
          IntegerAttr::get(i32Ty(ctx), intAttr.getInt()));

    if (auto strAttr = llvm::dyn_cast<StringAttr>(arg)) {
      // argc(<list>) arrives as "%argc:<name>": a list length known only now.
      llvm::StringRef s = strAttr.getValue();
      // A real null pointer, not the unknown-token undef fallback.
      if (s == "null")
        return LLVM::ZeroOp::create(builder, loc, ptrTy(ctx));
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

    return LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
  }

  void lowerCall(PlanCallAttr callAttr) {
    ArrayAttr argAttrs = callAttr.getArgs();
    SmallVector<Value> args;
    SmallVector<Type>  argTypes;
    // Where the first spliced list started, i.e. how many args are fixed.  A
    // call that splices a list is variadic: capture counts differ per call site.
    std::optional<size_t> fixedArgs;
    args.reserve(argAttrs.size());
    argTypes.reserve(argAttrs.size());
    for (Attribute a : argAttrs) {
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
      // As a call argument the if clause is a C _Bool; as a branch condition it
      // stays i1, which is why this widening is here and not in resolveArg.
      if (auto s = llvm::dyn_cast<StringAttr>(a))
        if (s.getValue() == "if_clause") v = toI8(v);
      args.push_back(v);
      argTypes.push_back(v.getType());
    }

    llvm::StringRef callee = callAttr.getCallee().getValue();
    StringAttr resultName = callAttr.getResult();

    // A callee naming a binding is a function *value*, not a symbol: the
    // outlined function's real name is known only to the outlining pass, which
    // hands its address over as `body`.  Called through the pointer, typed void
    // — an entry that does return something is called for its effect only.
    // Checked before the `let` case so a bound callee stays indirect.
    if (auto bound = bindings.find(callee); bound != bindings.end()) {
      if (resultName)
        emitError(loc, "omp-lower-plan: `let ... = call " + callee +
                       "(...)` binds the result of an indirect call, which is "
                       "not supported (the callee's return type is unknown "
                       "here)");
      auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                              argTypes, /*isVarArg=*/false);
      SmallVector<Value> operands;
      operands.push_back(bound->second);
      operands.append(args.begin(), args.end());
      LLVM::CallOp::create(builder, loc, fnTy, operands);
      return;
    }

    // `let <name> = call ...` yields a handle the rest of the plan refers to as
    // "%<name>" — the kmp_task_t the alloc returns.  Entry points that hand back
    // a handle return a pointer; those whose result the plan drops are declared
    // void below.
    if (resultName) {
      func::FuncOp decl = getOrInsertDeclWithReturn(module, callee, argTypes,
                                                    ptrTy(ctx), builder);
      Value res = func::CallOp::create(builder, loc, decl, args).getResult(0);
      bindings["%" + resultName.getValue().str()] = res;
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

  // `emit populate_shareds(<task>)`: write the captures into the block the
  // runtime allocated alongside the task, reached as load(task->shareds).  The
  // outlining pass already resolved what each field receives and handed the
  // values over as %captures, so the shareds struct is one field per bound
  // value, in order — exactly as the entry prolog rebuilds it.
  void lowerPopulateShareds(PlanEmitAttr emitAttr) {
    // Skipping the write would leave the task reading uninitialised shareds —
    // a miscompile with no other symptom.
    if (!kmpTaskTy) {
      emitError(loc, "omp-lower-plan: `emit populate_shareds` needs the "
                     "construct's `kmp_task_t = struct(...)` layout, which this "
                     "runtime does not declare");
      return;
    }
    auto it = listBindings.find("%captures");
    if (it == listBindings.end()) {
      emitError(loc, "omp-lower-plan: `emit populate_shareds` found no "
                     "`%captures` list; the outlining pass did not bind one");
      return;
    }
    // No captures is not an error: there is simply nothing to write.
    if (it->second.empty()) return;
    ArrayRef<Value> capVals = it->second;

    Value base = resolveArg(emitAttr.getValue());
    auto ptr = ptrTy(ctx);
    Value sg = LLVM::GEPOp::create(builder, loc, ptr, kmpTaskTy, base,
                                   ArrayRef<LLVM::GEPArg>{0, 0});
    Value sh = LLVM::LoadOp::create(builder, loc, ptr, sg);

    SmallVector<Type> fieldTypes;
    for (Value v : capVals) fieldTypes.push_back(v.getType());
    auto sharedsTy = LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);

    for (auto [i, v] : llvm::enumerate(capVals)) {
      Value gep = LLVM::GEPOp::create(builder, loc, ptr, sharedsTy, sh,
                                      ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
      LLVM::StoreOp::create(builder, loc, v, gep);
    }
  }

  // Every string token under this block, as a call argument or branch condition.
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

  // `branch <cond> { true => ... false => ... }`: the one plan action that is
  // not a straight line.  Splits the block and fills one per arm.
  void lowerBranch(PlanBranchAttr br) {
    Value cond = clauseToI1(builder, loc, resolveArg(br.getCond()));

    // Materialised before the split: left to the arms, whichever referenced one
    // first would define it where the other cannot see it.  Only if an arm asks.
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
        if (nm == "populate_shareds") {
          lowerPopulateShareds(emitAttr);
          continue;
        }
        // `emit` names a C++-backed verb.  The wsloop ones (loop_body,
        // thread_bounds) are consumed by the outlining pass and never arrive
        // here, so anything else is a verb this pass does not implement —
        // silently binding it to undef would hide a DSL typo.
        emitError(loc, "omp-lower-plan: unknown `emit " + nm + "`");
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
};

static llvm::StringRef getPropStr(ConstructOp op, llvm::StringRef key) {
  auto dict = op.getPropDict();
  if (!dict) return {};
  if (auto sa = llvm::dyn_cast_or_null<StringAttr>(dict.get(key)))
    return sa.getValue();
  return {};
}

// A plain walk rather than a dialect conversion: `branch` splits the block it is
// emitted into, which the conversion framework's change tracking does not fit.
static LogicalResult lowerConstruct(ConstructOp op) {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op.emitOpError("expected parent ModuleOp");

    OpBuilder builder(op);
    builder.setInsertionPoint(op);

    ConstructEmitter emitter{module, builder, op.getLoc(), op.getContext()};

    // DSL-owned, arriving as the same "%struct:..." token the outlining pass
    // expands.  Only field 0, the shareds pointer, is read here.
    if (llvm::StringRef layout = getPropStr(op, "kmp_task_t"); !layout.empty())
      emitter.kmpTaskTy =
          parseStructProp(op.getContext(), layout, LLVM::LLVMStructType());

    // Seed the bindings from the construct's operands, named 1:1 by the names
    // array.  Bindings are consulted first, so a %gtid bound by the outlining
    // pass wins over materialising a fresh one.  Names declared as lists get an
    // entry even when no operand carries them, so an empty list stays a list:
    // `captures` on a parallel that captures nothing must splice into zero
    // arguments, not fall through to undef.
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

    emitter.lowerActions(op.getPre());
    emitter.lowerActions(op.getInvoke());
    emitter.lowerActions(op.getPost());

    op.erase();
    return success();
}

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
