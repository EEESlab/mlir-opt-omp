// OmpOutliningPass.cpp
//
// Two responsibilities in one pass:
//
// 1. PARALLEL OUTLINING — for each omp_lower.construct with a body region:
//    - Collect captures, create @outlined_parallel_N func.func
//    - Move region body into it, wire captured values as block args
//    - Emit __kmpc_fork_call (iomp) or GOMP_parallel (libgomp) with captures
//    - Create __omp_ident_N global if DSL pre block contains "emit ident"
//
//    Capture strategies:
//      by_pointer (iomp): each capture passed as a separate pointer argument
//      packed (libgomp):  all captures packed into an alloca'd struct,
//                         a single ptr to the struct is passed as 'data'
//
// 2. WSLOOP LOWERING — for each omp.wsloop surviving inside outlined funcs:
//    - Extract context (schedule, nowait, bounds) from the omp.loop_nest
//    - Call dsl::Evaluator::buildPlan(runtime, "wsloop", ctx) to get the plan
//    - Emit runtime calls and an explicit loop, driven by the plan's invoke block

#include "OmpOutliningPass.h"
#include "OmpLoweringOps.h"
#include "DSLEvaluator.h"
#include "DSLParser.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;
using namespace mlir::omp_lower;

namespace {

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

static Type ptrTy(MLIRContext *ctx) {
  return LLVM::LLVMPointerType::get(ctx);
}

static Type i32Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 32);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Classify captures: returns true if this capture is a private variable
// (scalar alloca written before read inside the region — e.g. loop IV).
static bool isPrivateCapture(Value val, Region &region) {
  if (!val.getDefiningOp<LLVM::AllocaOp>()) return false;
  auto allocaOp = val.getDefiningOp<LLVM::AllocaOp>();
  Type elemTy = allocaOp.getElemType();
  // Only scalar non-pointer types can be private IVs.
  if (!LLVM::isCompatibleType(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMPointerType>(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMArrayType>(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMStructType>(elemTy)) return false;
  // Check if first use in region is a store (addr operand).
  bool firstIsStore = false, found = false;
  region.walk([&](Operation *op) {
    if (found) return;
    for (auto &use : op->getOpOperands()) {
      if (use.get() == val) {
        found = true;
        if (auto st = llvm::dyn_cast<LLVM::StoreOp>(op))
          if (&use == &op->getOpOperand(1)) firstIsStore = true;
        break;
      }
    }
  });
  return firstIsStore;
}

static SmallVector<Value> collectCaptures(Region &region) {
  llvm::SetVector<Value> seen;
  SmallVector<Value> captures;
  region.walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      if (!region.isAncestor(operand.getParentRegion()) && seen.insert(operand))
        captures.push_back(operand);
  });
  return captures;
}

static void replaceUsesInRegion(Region &region, Value oldVal, Value newVal) {
  for (auto &use : llvm::make_early_inc_range(oldVal.getUses()))
    if (region.isAncestor(use.getOwner()->getParentRegion()))
      use.set(newVal);
}

static std::string getPropStr(ConstructOp op, llvm::StringRef key) {
  auto dict = op.getPropDict();
  if (!dict) return "";
  if (auto sa = llvm::dyn_cast_or_null<StringAttr>(dict.get(key)))
    return sa.getValue().str();
  return "";
}

static func::FuncOp getOrInsertDecl(ModuleOp module,
                                    llvm::StringRef name,
                                    ArrayRef<Type> argTypes,
                                    OpBuilder &builder) {
  if (auto f = module.lookupSymbol<func::FuncOp>(name)) return f;
  auto fnType = builder.getFunctionType(argTypes, {});
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto decl = func::FuncOp::create(builder.getUnknownLoc(), name, fnType);
  module.getBody()->push_back(decl);
  decl.setVisibility(SymbolTable::Visibility::Private);
  decl->setAttr("llvm.linkage",
                LLVM::LinkageAttr::get(module.getContext(),
                                       LLVM::Linkage::External));
  return decl;
}

static func::FuncOp getOrInsertDeclWithReturn(ModuleOp module,
                                              llvm::StringRef name,
                                              ArrayRef<Type> argTypes,
                                              Type returnType,
                                              OpBuilder &builder) {
  if (auto f = module.lookupSymbol<func::FuncOp>(name)) return f;
  SmallVector<Type> resultTypes = {returnType};
  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto decl = func::FuncOp::create(builder.getUnknownLoc(), name, fnType);
  module.getBody()->push_back(decl);
  decl.setVisibility(SymbolTable::Visibility::Private);
  decl->setAttr("llvm.linkage",
                LLVM::LinkageAttr::get(module.getContext(),
                                       LLVM::Linkage::External));
  return decl;
}

// Emit a call to a no-arg function returning i32 (e.g. pi_core_id).
// Handles both func.func and llvm.func declarations.
static Value emitNoArgI32Call(ModuleOp module, OpBuilder &builder,
                               Location loc, llvm::StringRef name) {
  MLIRContext *ctx = module.getContext();
  Type i32t = IntegerType::get(ctx, 32);
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return LLVM::CallOp::create(builder, loc, i32t, name,
                                 ValueRange{}).getResult();
  }
  auto decl = getOrInsertDeclWithReturn(module, name, {}, i32t, builder);
  return func::CallOp::create(builder, loc, decl, ValueRange{}).getResult(0);
}

// Get the address of an existing __omp_ident_ global, or undef if none.
static Value getIdentAddr(ModuleOp module, OpBuilder &builder, Location loc,
                          MLIRContext *ctx) {
  for (auto &op : *module.getBody())
    if (auto g = llvm::dyn_cast<LLVM::GlobalOp>(op))
      if (g.getSymName().starts_with("__omp_ident_"))
        return LLVM::AddressOfOp::create(builder, loc, ptrTy(ctx),
          FlatSymbolRefAttr::get(ctx, g.getSymName()));
  return LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
}

// ---------------------------------------------------------------------------
// 1. PARALLEL OUTLINING
// ---------------------------------------------------------------------------

static void outlineConstruct(ConstructOp op, ModuleOp module, int &counter,
                              llvm::StringRef runtimeName) {
  Region &body = op.getBody();
  if (body.empty()) return;

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();

  std::string outlineSig   = getPropStr(op, "outline_signature");
  std::string captureStrat = getPropStr(op, "capture_strategy");

  bool isMicrotask = outlineSig.find("microtask") != std::string::npos;
  bool isClosure   = outlineSig.find("closure")   != std::string::npos;
  bool isPacked    = captureStrat == "packed";

  // Collect all values used inside the region but defined outside.
  SmallVector<Value> captures = collectCaptures(body);

  // Compute private captures before takeBody (body region is consumed after).
  llvm::SetVector<Value> privateCaptures;
  for (auto cap : captures)
    if (isPrivateCapture(cap, body))
      privateCaptures.insert(cap);

  std::string fnName = "outlined_parallel_" + std::to_string(counter++);

  // Build outlined function argument types.
  SmallVector<Type> fnArgTypes;
  if (isMicrotask) {
    fnArgTypes.push_back(ptrTy(ctx)); // ptr gtid
    fnArgTypes.push_back(ptrTy(ctx)); // ptr btid
    if (isPacked)
      fnArgTypes.push_back(ptrTy(ctx)); // ptr to capture struct
    else
      for (auto cap : captures) fnArgTypes.push_back(cap.getType());
  } else if (isClosure) {
    // libgomp: void(void *data) — data points to capture struct
    fnArgTypes.push_back(ptrTy(ctx));
  } else {
    for (auto cap : captures) fnArgTypes.push_back(cap.getType());
  }

  OpBuilder builder(ctx);
  if (auto parentFn = op->getParentOfType<func::FuncOp>())
    builder.setInsertionPoint(parentFn);
  else
    builder.setInsertionPointToStart(module.getBody());

  auto outlinedFn = func::FuncOp::create(loc, fnName,
    FunctionType::get(ctx, fnArgTypes, {}));
  outlinedFn.setPrivate();
  builder.insert(outlinedFn);

  // Strip omp.wsloop/omp.parallel private_vars operands before takeBody to
  // prevent replaceUsesInRegion from corrupting those operand references.
  body.walk([&](Operation *walkOp) {
    if (auto wsOp = llvm::dyn_cast<omp::WsloopOp>(walkOp))
      if (!wsOp.getPrivateVars().empty())
        wsOp.getPrivateVarsMutable().clear();
    if (auto parOp = llvm::dyn_cast<omp::ParallelOp>(walkOp))
      if (!parOp.getPrivateVars().empty())
        parOp.getPrivateVarsMutable().clear();
  });

  // Take the body. Entry block may have privatizer args from omp.parallel.
  outlinedFn.getBody().takeBody(body);
  Block &entry = outlinedFn.getBody().front();

  // Save existing privatizer block args before inserting capture args.
  SmallVector<BlockArgument> privatizerArgs(entry.getArguments().begin(),
                                             entry.getArguments().end());

  if (isPacked) {
    // -------------------------------------------------------------------------
    // PACKED / CLOSURE strategy (libgomp)
    // -------------------------------------------------------------------------
    // The outlined function receives a single ptr to a capture struct.
    // We unpack each capture from the struct in the function prolog.
    //
    // Struct layout: { type_0, type_1, ..., type_N-1 }
    // Each field is the capture value itself (not a pointer to it) for
    // scalar types, or a pointer for pointer types.
    //
    // For isMicrotask+isPacked (unusual): prepend gtid/btid first.
    if (isMicrotask) {
      entry.insertArgument(0u, ptrTy(ctx), loc); // btid
      entry.insertArgument(0u, ptrTy(ctx), loc); // gtid
    }
    // Always insert the data ptr at position 0 (closure) or 2 (microtask).
    // The entry block may already have privatizer args — we insert before them.
    unsigned dataPtrIdx = isMicrotask ? 2u : 0u;
    entry.insertArgument(dataPtrIdx, ptrTy(ctx), loc);
    BlockArgument dataPtr = entry.getArgument(dataPtrIdx);

    // Build the capture struct type: { cap_0_type, cap_1_type, ... }
    SmallVector<Type> fieldTypes;
    for (auto cap : captures) fieldTypes.push_back(cap.getType());
    auto structTy = LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);

    // In the function prolog, unpack each capture from the struct.
    if (!captures.empty()) {
      OpBuilder prologue(&entry, entry.begin());
      Value one64 = LLVM::ConstantOp::create(prologue, loc,
        IntegerType::get(ctx, 64),
        IntegerAttr::get(IntegerType::get(ctx, 64), 1));
      // Keep loaded values so privatizer handling can reuse them.
      SmallVector<Value> loadedCaptures;
      for (size_t i = 0; i < captures.size(); i++) {
        if (privateCaptures.contains(captures[i])) {
          // Private capture (loop IV etc.): allocate fresh alloca per core.
          // Don't load from struct — each core gets its own independent copy.
          auto srcAlloca = captures[i].getDefiningOp<LLVM::AllocaOp>();
          Value freshAlloca = LLVM::AllocaOp::create(prologue, loc,
            ptrTy(ctx), srcAlloca.getElemType(), one64);
          loadedCaptures.push_back(freshAlloca);
          replaceUsesInRegion(outlinedFn.getBody(), captures[i], freshAlloca);
        } else {
          Value gep = LLVM::GEPOp::create(prologue, loc, ptrTy(ctx), structTy,
            dataPtr, ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
          Value loaded = LLVM::LoadOp::create(prologue, loc,
            captures[i].getType(), gep);
          loadedCaptures.push_back(loaded);
          replaceUsesInRegion(outlinedFn.getBody(), captures[i], loaded);
        }
      }
      // Handle privatizer args (firstprivate).
      // We already loaded the source ptrs from the struct above and stored
      // them in loadedCaptures[i]. Reuse those values — don't create new GEPs.
      size_t numPriv = privatizerArgs.size();
      // The injected unrealized_conversion_cast ops for privatizer sources
      // are inserted at the START of the entry block, so collectCaptures
      // finds them FIRST — they are at indices 0..numPriv-1.
      size_t privCapStart = 0;
      for (size_t i = 0; i < numPriv; i++) {
        BlockArgument dstArg = privatizerArgs[i];
        Value srcPtr = loadedCaptures[privCapStart + i]; // already loaded ptr
        // Find element type from loads of dstArg.
        Type elemTy;
        outlinedFn.getBody().walk([&](LLVM::LoadOp loadOp) {
          if (!elemTy && loadOp.getAddr() == dstArg)
            elemTy = loadOp.getRes().getType();
        });
        if (!elemTy || !LLVM::isCompatibleType(elemTy)) continue;
        if (!llvm::isa<LLVM::LLVMPointerType>(srcPtr.getType())) continue;
        Value privAlloca = LLVM::AllocaOp::create(prologue, loc,
          ptrTy(ctx), elemTy, one64);
        Value srcVal = LLVM::LoadOp::create(prologue, loc, elemTy, srcPtr);
        LLVM::StoreOp::create(prologue, loc, srcVal, privAlloca);
        replaceUsesInRegion(outlinedFn.getBody(), dstArg, privAlloca);
      }
    }
  } else if (isMicrotask) {
    // -------------------------------------------------------------------------
    // BY_POINTER strategy (iomp microtask)
    // -------------------------------------------------------------------------
    // Insert one capture arg per capture in reverse order.
    for (int i = (int)captures.size() - 1; i >= 0; i--) {
      BlockArgument arg = entry.insertArgument(0u, captures[i].getType(), loc);
      replaceUsesInRegion(outlinedFn.getBody(), captures[i], arg);
    }
    // Prepend btid then gtid.
    entry.insertArgument(0u, ptrTy(ctx), loc); // btid
    entry.insertArgument(0u, ptrTy(ctx), loc); // gtid

    // Emit firstprivate copies in the function prolog.
    if (!privatizerArgs.empty()) {
      size_t numPriv = privatizerArgs.size();
      size_t privCapStart = 2; // captures injected by OmpToOmpLowerPass start here
      OpBuilder copyBuilder(&entry, entry.begin());
      Value one64 = LLVM::ConstantOp::create(copyBuilder, loc,
        IntegerType::get(ctx, 64),
        IntegerAttr::get(IntegerType::get(ctx, 64), 1));
      for (size_t i = 0; i < numPriv; i++) {
        BlockArgument srcArg = entry.getArgument(privCapStart + i);
        BlockArgument dstArg = privatizerArgs[i];
        // Find element type from existing loads of dstArg.
        Type elemTy;
        outlinedFn.getBody().walk([&](LLVM::LoadOp loadOp) {
          if (!elemTy && loadOp.getAddr() == dstArg)
            elemTy = loadOp.getRes().getType();
        });
        if (!elemTy) {
          // Fallback: try original capture alloca's elem type.
          if (i < captures.size())
            if (auto alloca = captures[i].getDefiningOp<LLVM::AllocaOp>())
              elemTy = alloca.getElemType();
        }
        if (!elemTy || !LLVM::isCompatibleType(elemTy)) continue;
        if (!llvm::isa<LLVM::LLVMPointerType>(srcArg.getType())) continue;
        Value privAlloca = LLVM::AllocaOp::create(copyBuilder, loc,
          ptrTy(ctx), elemTy, one64);
        Value srcVal = LLVM::LoadOp::create(copyBuilder, loc, elemTy, srcArg);
        LLVM::StoreOp::create(copyBuilder, loc, srcVal, privAlloca);
        replaceUsesInRegion(outlinedFn.getBody(), dstArg, privAlloca);
        // Fix alignment on loads/stores of the private alloca.
        // The original privatizer arg loads used alignment=1; fix to natural align.
        if (LLVM::isCompatibleType(elemTy)) {
          unsigned naturalAlign = 1;
          if (elemTy.isInteger(32) || elemTy.isF32()) naturalAlign = 4;
          else if (elemTy.isInteger(64) || elemTy.isF64()) naturalAlign = 8;
          else if (elemTy.isInteger(16)) naturalAlign = 2;
          if (naturalAlign > 1) {
            outlinedFn.getBody().walk([&](Operation *useOp) {
              if (auto load = llvm::dyn_cast<LLVM::LoadOp>(useOp)) {
                if (load.getAddr() == privAlloca &&
                    load.getAlignment().value_or(0) < naturalAlign)
                  load.setAlignment(naturalAlign);
              }
              if (auto store = llvm::dyn_cast<LLVM::StoreOp>(useOp)) {
                if (store.getAddr() == privAlloca &&
                    store.getAlignment().value_or(0) < naturalAlign)
                  store.setAlignment(naturalAlign);
              }
            });
          }
        }
      }
    }
  } else if (isClosure) {
    // closure without microtask: single env ptr already at index 0
  }

  // Remove unused privatizer block args from the entry block.
  // They were already replaced by private copies in the prolog; keeping them
  // would add extra parameters that receive no values at the call site.
  if (!privatizerArgs.empty()) {
    for (auto arg : llvm::reverse(privatizerArgs))
      if (arg.use_empty())
        entry.eraseArgument(arg.getArgNumber());
  }

  // Remove injected unrealized_conversion_cast marker ops (no users).
  SmallVector<Operation *> injectedCasts;
  outlinedFn.getBody().walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->use_empty()) injectedCasts.push_back(castOp);
  });
  for (auto *c : injectedCasts) c->erase();

  // Erase unused capture args from the entry block and filter captures list.
  // This removes captures that were only needed as privatizer sources
  // (e.g., source allocas for private vars that don't need copying).
  {
    unsigned capBase = isMicrotask ? 2 : (isClosure ? 1 : 0);
    llvm::DenseSet<unsigned> erasedCapIdx;
    for (int i = (int)captures.size() - 1; i >= 0; i--) {
      unsigned argIdx = capBase + (unsigned)i;
      if (argIdx < entry.getNumArguments() &&
          entry.getArgument(argIdx).use_empty()) {
        entry.eraseArgument(argIdx);
        erasedCapIdx.insert((unsigned)i);
      }
    }
    if (!erasedCapIdx.empty()) {
      SmallVector<Value> filtered;
      for (size_t i = 0; i < captures.size(); i++)
        if (!erasedCapIdx.count(i))
          filtered.push_back(captures[i]);
      captures = std::move(filtered);
    }
  }

  // Update function type to match actual entry block args.
  SmallVector<Type> finalArgTypes;
  for (auto arg : entry.getArguments())
    finalArgTypes.push_back(arg.getType());
  outlinedFn.setFunctionType(FunctionType::get(ctx, finalArgTypes, {}));

  // Replace omp.terminator with func.return (direct blocks only).
  SmallVector<Operation *> terminators;
  for (auto &block : outlinedFn.getBody())
    for (auto &innerOp : block)
      if (innerOp.getName().getStringRef() == "omp.terminator")
        terminators.push_back(&innerOp);
  for (auto *termOp : terminators) {
    OpBuilder tb(termOp);
    func::ReturnOp::create(tb, termOp->getLoc());
    termOp->erase();
  }

  // Lower omp.barrier inside the outlined function — runtime-aware.
  SmallVector<Operation *> barriers;
  for (auto &block : outlinedFn.getBody())
    for (auto &innerOp : block)
      if (innerOp.getName().getStringRef() == "omp.barrier")
        barriers.push_back(&innerOp);
  for (auto *barrierOp : barriers) {
    OpBuilder bb(barrierOp);
    if (runtimeName == "libgomp") {
      auto decl = getOrInsertDecl(module, "GOMP_barrier", {}, bb);
      func::CallOp::create(bb, barrierOp->getLoc(), decl, ValueRange{});
    } else if (runtimeName == "pmsis") {
      auto decl = getOrInsertDecl(module, "ext_pi_cl_team_barrier", {}, bb);
      func::CallOp::create(bb, barrierOp->getLoc(), decl, ValueRange{});
    } else {
      // __kmpc_barrier(ident, gtid)
      Value identAddr = getIdentAddr(module, bb, barrierOp->getLoc(), ctx);
      Value gtidVal = LLVM::UndefOp::create(bb, barrierOp->getLoc(), i32Ty(ctx));
      auto &fnEntry = outlinedFn.getBody().front();
      if (fnEntry.getNumArguments() >= 2)
        gtidVal = LLVM::LoadOp::create(bb, barrierOp->getLoc(), i32Ty(ctx),
                                        fnEntry.getArgument(0));
      SmallVector<Type> bt = {ptrTy(ctx), i32Ty(ctx)};
      auto decl = getOrInsertDecl(module, "__kmpc_barrier", bt, bb);
      func::CallOp::create(bb, barrierOp->getLoc(), decl,
                            ValueRange{identAddr, gtidVal});
    }
    barrierOp->erase();
  }

  // ---- Check DSL pre block for emit ident / emit global_tid ----
  bool needsIdent     = false;
  bool needsGlobalTid = false;
  for (auto attr : op.getPre()) {
    if (auto ea = llvm::dyn_cast<PlanEmitAttr>(attr)) {
      if (ea.getSymName().getValue() == "ident")      needsIdent = true;
      if (ea.getSymName().getValue() == "global_tid") needsGlobalTid = true;
    }
  }

  std::string identName = "__omp_ident_" + std::to_string(counter - 1);
  if (needsIdent && !module.lookupSymbol(identName)) {
    auto i32t = IntegerType::get(ctx, 32);
    auto identStructTy = LLVM::LLVMStructType::getLiteral(
      ctx, {i32t, i32t, i32t, i32t, ptrTy(ctx)});
    OpBuilder gBuilder(ctx);
    gBuilder.setInsertionPointToStart(module.getBody());
    auto global = LLVM::GlobalOp::create(gBuilder, loc,
      identStructTy, true, LLVM::Linkage::Private, identName, Attribute{});
    Block *initBlock = new Block();
    global.getInitializerRegion().push_back(initBlock);
    OpBuilder initB(ctx);
    initB.setInsertionPointToStart(initBlock);
    Value s = LLVM::UndefOp::create(initB, loc, identStructTy);
    auto ci = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(initB, loc, i32t,
        IntegerAttr::get(i32t, v));
    };
    auto ins = [&](Value v, Value st, unsigned idx) -> Value {
      return LLVM::InsertValueOp::create(initB, loc, identStructTy, st, v,
        ArrayRef<int64_t>{(int64_t)idx});
    };
    s = ins(ci(0), s, 0);
    s = ins(ci(2), s, 1);
    s = ins(ci(0), s, 2);
    s = ins(ci(0), s, 3);
    Value zero64 = LLVM::ConstantOp::create(initB, loc,
      IntegerType::get(ctx, 64), IntegerAttr::get(IntegerType::get(ctx,64),0));
    Value nullPtr = LLVM::IntToPtrOp::create(initB, loc, ptrTy(ctx), zero64);
    s = LLVM::InsertValueOp::create(initB, loc, identStructTy, s, nullPtr,
      ArrayRef<int64_t>{4});
    LLVM::ReturnOp::create(initB, loc, s);
  }

  // ---- Emit the runtime call at the call site ----
  std::string runtimeCallee;
  for (auto attr : op.getInvoke())
    if (auto ca = llvm::dyn_cast<PlanCallAttr>(attr)) {
      runtimeCallee = ca.getCallee().getValue().str();
      break;
    }

  if (!runtimeCallee.empty()) {
    builder.setInsertionPoint(op);

    SmallVector<Value> callArgs;
    SmallVector<Type>  callTypes;

    // ident
    Value identVal;
    if (needsIdent)
      identVal = LLVM::AddressOfOp::create(builder, loc, ptrTy(ctx),
        FlatSymbolRefAttr::get(ctx, identName));
    else
      identVal = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));

    // global_tid at call site
    Value gtidAtCallSite;
    if (needsGlobalTid) {
      auto gtidDecl = getOrInsertDeclWithReturn(module,
        "__kmpc_global_thread_num", {ptrTy(ctx)}, i32Ty(ctx), builder);
      gtidAtCallSite = func::CallOp::create(builder, loc, gtidDecl,
        ValueRange{identVal}).getResult(0);
    }

    // Emit pre-block calls (push_num_threads, push_proc_bind, etc.)
    for (auto attr : op.getPre()) {
      auto ca = llvm::dyn_cast<PlanCallAttr>(attr);
      if (!ca) continue;
      SmallVector<Value> preArgs;
      SmallVector<Type>  preTypes;
      for (auto argAttr : ca.getArgs()) {
        Value v;
        if (auto sa = llvm::dyn_cast<StringAttr>(argAttr)) {
          llvm::StringRef s = sa.getValue();
          if (s == "ident" || s == "%ident")
            v = identVal;
          else if (s == "global_tid" || s == "%tid")
            v = gtidAtCallSite
              ? gtidAtCallSite
              : LLVM::UndefOp::create(builder, loc, i32Ty(ctx));
          else
            v = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
        } else if (auto ia = llvm::dyn_cast<IntegerAttr>(argAttr)) {
          v = arith::ConstantOp::create(builder, loc, i32Ty(ctx),
            IntegerAttr::get(i32Ty(ctx), ia.getInt()));
        } else {
          v = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
        }
        preArgs.push_back(v); preTypes.push_back(v.getType());
      }
      auto preDecl = getOrInsertDecl(module, ca.getCallee().getValue(),
                                      preTypes, builder);
      func::CallOp::create(builder, loc, preDecl, preArgs);
    }

    if (isPacked) {
      // -----------------------------------------------------------------------
      // PACKED / CLOSURE: build capture struct on the stack, pass ptr to it
      // -----------------------------------------------------------------------
      SmallVector<Type> fieldTypes;
      for (auto cap : captures) fieldTypes.push_back(cap.getType());

      // Build the capture struct (same for all packed runtimes).
      Value structAlloca;
      if (!captures.empty()) {
        auto structTy = LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);
        Value one64 = LLVM::ConstantOp::create(builder, loc,
          IntegerType::get(ctx, 64),
          IntegerAttr::get(IntegerType::get(ctx, 64), 1));
        structAlloca = LLVM::AllocaOp::create(builder, loc,
          ptrTy(ctx), structTy, one64);
        for (size_t i = 0; i < captures.size(); i++) {
          Value capVal = captures[i];
          // For private captures (loop IVs), allocate a fresh local alloca
          // so each core gets its own private copy (no data race).
          if (privateCaptures.contains(capVal)) {
            // Private capture — slot not read in outlined fn; pass undef.
            capVal = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
          }
          Value gep = LLVM::GEPOp::create(builder, loc, ptrTy(ctx), structTy,
            structAlloca, ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
          LLVM::StoreOp::create(builder, loc, capVal, gep);
        }
      } else {
        structAlloca = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
      }

      // outlined fn pointer
      Value fnPtr = func::ConstantOp::create(builder, loc,
        outlinedFn.getFunctionType(),
        FlatSymbolRefAttr::get(ctx, fnName));
      Value fnPtrCast = UnrealizedConversionCastOp::create(
        builder, loc, TypeRange{ptrTy(ctx)}, ValueRange{fnPtr}).getResult(0);

      // Build call args from DSL invoke args, resolving symbolic names:
      //   "body"    → fnPtrCast  (the outlined function pointer)
      //   "env_ptr" → structAlloca (the capture struct pointer)
      //   integer   → i32 constant
      //   other str → undef ptr
      for (auto attr : op.getInvoke()) {
        auto ca = llvm::dyn_cast<PlanCallAttr>(attr);
        if (!ca) continue;
        for (auto argAttr : ca.getArgs()) {
          Value v;
          if (auto sa = llvm::dyn_cast<StringAttr>(argAttr)) {
            llvm::StringRef s = sa.getValue();
            if (s == "body" || s == "outlined_parallel")
              v = fnPtrCast;
            else if (s == "env_ptr")
              v = structAlloca;
            else
              v = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
          } else if (auto ia = llvm::dyn_cast<IntegerAttr>(argAttr)) {
            v = arith::ConstantOp::create(builder, loc, i32Ty(ctx),
              IntegerAttr::get(i32Ty(ctx), ia.getInt()));
          } else if (auto aa = llvm::dyn_cast<ArrayAttr>(argAttr)) {
            // Nested array — skip (handles argc(captures) style args)
            v = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
          } else {
            v = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
          }
          callArgs.push_back(v);
          callTypes.push_back(v.getType());
        }
        break; // only first invoke call
      }
    } else {
      // -----------------------------------------------------------------------
      // BY_POINTER (iomp): ident, argc, fn_ptr, captures...
      // -----------------------------------------------------------------------
      callArgs.push_back(identVal); callTypes.push_back(ptrTy(ctx));

      Value argcVal = arith::ConstantOp::create(builder, loc,
        builder.getI32Type(),
        builder.getI32IntegerAttr((int32_t)captures.size()));
      callArgs.push_back(argcVal); callTypes.push_back(i32Ty(ctx));

      Value fnPtr = func::ConstantOp::create(builder, loc,
        outlinedFn.getFunctionType(),
        FlatSymbolRefAttr::get(ctx, fnName));
      Value fnPtrCast = UnrealizedConversionCastOp::create(
        builder, loc, TypeRange{ptrTy(ctx)}, ValueRange{fnPtr}).getResult(0);
      callArgs.push_back(fnPtrCast); callTypes.push_back(ptrTy(ctx));

      for (auto cap : captures) {
        Value capVal = cap;
        // For private captures (loop IVs), pass a fresh local alloca.
        if (privateCaptures.contains(cap)) {
          // Private capture — pass a fresh local alloca so each core has
          // its own copy (not shared through the original caller alloca).
          auto srcAlloca = cap.getDefiningOp<LLVM::AllocaOp>();
          Value cnt = LLVM::ConstantOp::create(builder, loc,
            IntegerType::get(ctx, 64),
            IntegerAttr::get(IntegerType::get(ctx, 64), 1));
          capVal = LLVM::AllocaOp::create(builder, loc, ptrTy(ctx),
            srcAlloca.getElemType(), cnt);
        }
        callArgs.push_back(capVal); callTypes.push_back(capVal.getType());
      }
    }

    // __kmpc_fork_call is variadic: (ident, argc, fn, ...captures) -> void.
    // Declare it as llvm.func variadic so multiple parallel regions with
    // different capture counts can share the same declaration.
    if (runtimeCallee == "__kmpc_fork_call" ||
        runtimeCallee == "__kmpc_fork_call_if") {
      MLIRContext *mctx = module.getContext();
      // Fixed prefix: ident(ptr), argc(i32), fn(ptr)
      SmallVector<Type> fixedTypes;
      size_t fixedCount = std::min((size_t)3, callTypes.size());
      for (size_t fi = 0; fi < fixedCount; fi++)
        fixedTypes.push_back(callTypes[fi]);
      LLVM::LLVMFuncOp llvmDecl;
      if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(runtimeCallee)) {
        llvmDecl = existing;
      } else {
        auto varFnType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(mctx), fixedTypes, /*isVarArg=*/true);
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(module.getBody());
        llvmDecl = builder.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(mctx), runtimeCallee, varFnType,
          LLVM::Linkage::External);
      }
      LLVM::CallOp::create(builder, loc, llvmDecl, callArgs);
    } else {
      auto decl = getOrInsertDecl(module, runtimeCallee, callTypes, builder);
      func::CallOp::create(builder, loc, decl, callArgs);
    }
  }

  op.erase();
}

// ---------------------------------------------------------------------------
// 2. WSLOOP LOWERING — driven by DSL plan
// ---------------------------------------------------------------------------

static void lowerWsloop(omp::WsloopOp wsOp,
                        ModuleOp module,
                        const dsl::LoweringPlan &plan) {
  omp::LoopNestOp loopNest;
  wsOp.walk([&](omp::LoopNestOp op) { loopNest = op; });
  if (!loopNest) return;

  MLIRContext *ctx = wsOp.getContext();
  Location loc = wsOp.getLoc();
  OpBuilder builder(wsOp);

  auto lbs   = loopNest.getLoopLowerBounds();
  auto ubs   = loopNest.getLoopUpperBounds();
  auto steps = loopNest.getLoopSteps();
  if (lbs.empty() || ubs.empty() || steps.empty()) return;

  Value lb = lbs[0], ub = ubs[0], step = steps[0];
  Type iterTy = lb.getType();

  // Get ident and gtid from enclosing outlined function.
  // For iomp microtask: arg0 = ptr gtid, arg1 = ptr btid → load i32 from arg0.
  // For libgomp closure: arg0 = ptr data (capture struct) — no gtid ptr.
  //   Use omp_get_thread_num() to obtain the current thread id.
  Value identAddr = getIdentAddr(module, builder, loc, ctx);
  Value gtidVal   = LLVM::UndefOp::create(builder, loc, iterTy);
  if (auto parentFn = wsOp->getParentOfType<func::FuncOp>()) {
    auto &entry = parentFn.getBody().front();
    unsigned numArgs = entry.getNumArguments();
    // Microtask convention: at least 2 args (gtid ptr, btid ptr) + captures.
    // Closure convention: exactly 1 arg (data ptr).
    bool isMicrotaskFn = numArgs >= 2 &&
      llvm::isa<LLVM::LLVMPointerType>(entry.getArgument(0).getType()) &&
      llvm::isa<LLVM::LLVMPointerType>(entry.getArgument(1).getType());
    if (isMicrotaskFn) {
      // Load gtid from the first arg (ptr to i32).
      Value gtidPtr = entry.getArgument(0);
      gtidVal = LLVM::LoadOp::create(builder, loc, iterTy, gtidPtr);
    } else {
      // libgomp closure: call omp_get_thread_num() to get gtid.
      // If already declared as llvm.func, use llvm.call; otherwise
      // create a func.func declaration and use func.call.
      if (module.lookupSymbol<LLVM::LLVMFuncOp>("omp_get_thread_num")) {
        auto callOp = LLVM::CallOp::create(builder, loc, iterTy,
          "omp_get_thread_num", ValueRange{});
        gtidVal = callOp.getResult();
      } else {
        auto gtidDecl = getOrInsertDeclWithReturn(module, "omp_get_thread_num",
                                                   {}, iterTy, builder);
        gtidVal = func::CallOp::create(builder, loc, gtidDecl,
                                        ValueRange{}).getResult(0);
      }
    }
  }

  auto ptrT = ptrTy(ctx);
  Value one64 = LLVM::ConstantOp::create(builder, loc,
    IntegerType::get(ctx, 64), IntegerAttr::get(IntegerType::get(ctx,64), 1));
  Value zero32 = LLVM::ConstantOp::create(builder, loc, iterTy,
    IntegerAttr::get(iterTy, 0));
  Value one32 = LLVM::ConstantOp::create(builder, loc, iterTy,
    IntegerAttr::get(iterTy, 1));

  // Allocate plb, pub, pstride, plast.
  // plb/pub/plast are in/out: initialized here, overwritten by the runtime.
  // pstride is pure output: NOT initialized — runtime writes the stride value.
  Value plb     = LLVM::AllocaOp::create(builder, loc, ptrT, iterTy, one64);
  Value pub     = LLVM::AllocaOp::create(builder, loc, ptrT, iterTy, one64);
  Value pstride = LLVM::AllocaOp::create(builder, loc, ptrT, iterTy, one64);
  Value plast   = LLVM::AllocaOp::create(builder, loc, ptrT, iterTy, one64);

  // Convert exclusive upper bound to inclusive last-iteration index.
  // omp.loop_nest uses exclusive ub: trip = (ub - lb) / step iterations.
  // __kmpc_for_static_init_4 expects inclusive upper bound: ub_incl = lb + (trip-1)*step
  //   = lb + ((ub-lb)/step - 1)*step  = ub - step
  Value ubInclusive = LLVM::SubOp::create(builder, loc, ub, step);

  LLVM::StoreOp::create(builder, loc, lb,          plb);
  LLVM::StoreOp::create(builder, loc, ubInclusive, pub);
  // pstride: no initialization — pure output parameter
  LLVM::StoreOp::create(builder, loc, zero32,      plast);

  // Map symbolic DSL names to SSA values.
  auto resolveCallArg = [&](const dsl::Value &v) -> Value {
    if (auto *sv = std::get_if<dsl::StrVal>(&v)) {
      llvm::StringRef s = sv->value;
      if (s == "%ident" || s == "ident")      return identAddr;
      if (s == "%tid"   || s == "global_tid") return gtidVal;
      if (s == "%lb"    || s == "lower")      return plb;
      if (s == "%ub"    || s == "upper")      return pub;
      if (s == "%step"  || s == "step")       return step;    // actual loop step
      if (s == "%stride" || s == "stride")    return pstride; // output ptr for runtime
      if (s == "last"   || s == "plast")      return plast;
      return LLVM::UndefOp::create(builder, loc, ptrT);
    }
    if (auto *iv = std::get_if<dsl::IntVal>(&v))
      return LLVM::ConstantOp::create(builder, loc, iterTy,
        IntegerAttr::get(iterTy, iv->value));
    return LLVM::UndefOp::create(builder, loc, ptrT);
  };

  // Determine loop lowering strategy from DSL invoke callee names.
  bool isGOMP    = false;
  bool isPMSIS   = false;
  for (auto &action : plan.invoke) {
    if (auto *ca = std::get_if<dsl::PlanCall>(&action)) {
      if (ca->callee.find("GOMP_loop") != std::string::npos) isGOMP  = true;
      if (ca->callee == "core_bounds")                        isPMSIS = true;
    }
  }

  if (isPMSIS) {
    // -------------------------------------------------------------------------
    // PMSIS wsloop: static per-core bound computation, no runtime loop API.
    //
    // core_id   = pi_core_id()
    // num_cores = pi_cl_nb_cores()
    // chunk     = ceil((ub - lb) / step / num_cores)   [iterations per core]
    // lb_core   = lb + core_id * chunk * step
    // ub_core   = min(lb_core + chunk * step, ub)      [exclusive]
    //
    // Then a sequential loop from lb_core to ub_core step step.
    // Post-loop: optional pi_cl_team_barrier() if not nowait.
    // -------------------------------------------------------------------------
    Value coreId   = emitNoArgI32Call(module, builder, loc, "ext_pi_core_id");
    Value numCores = emitNoArgI32Call(module, builder, loc, "ext_pi_cl_nb_cores");

    // trip = (ub - lb + step - 1) / step  [ceiling division of iteration count]
    Value range    = LLVM::SubOp::create(builder, loc, ub, lb);
    Value rangeS   = LLVM::AddOp::create(builder, loc, range,
                       LLVM::SubOp::create(builder, loc, step, one32));
    Value trip     = LLVM::SDivOp::create(builder, loc, rangeS, step);

    // chunk = ceil(trip / num_cores)
    Value tripPlusNC = LLVM::AddOp::create(builder, loc, trip,
                         LLVM::SubOp::create(builder, loc, numCores, one32));
    Value chunk    = LLVM::SDivOp::create(builder, loc, tripPlusNC, numCores);

    // lb_core = lb + core_id * chunk * step
    Value coreChunk    = LLVM::MulOp::create(builder, loc, coreId, chunk);
    Value coreOff      = LLVM::MulOp::create(builder, loc, coreChunk, step);
    Value lbCore       = LLVM::AddOp::create(builder, loc, lb, coreOff);

    // ub_core = lb_core + chunk * step  (then clamp to ub)
    Value chunkStep    = LLVM::MulOp::create(builder, loc, chunk, step);
    Value lbCoreEnd    = LLVM::AddOp::create(builder, loc, lbCore, chunkStep);
    // clamp: if lbCoreEnd > ub then ub, else lbCoreEnd
    Value clampCond = LLVM::ICmpOp::create(builder, loc,
      LLVM::ICmpPredicate::sgt, lbCoreEnd, ub);
    // Use select: ub_core = clampCond ? ub : lbCoreEnd
    Value ubCore = LLVM::SelectOp::create(builder, loc, clampCond, ub, lbCoreEnd);

    // Build sequential loop: lb_core to ubCore (exclusive) step step.
    Block *preBlock   = builder.getInsertionBlock();
    Block *afterBlock = preBlock->splitBlock(builder.getInsertionPoint());
    Block *loopHeader = new Block();
    Block *loopBody   = new Block();
    Block *loopLatch  = new Block();

    auto &parentRegion = *preBlock->getParent();
    parentRegion.getBlocks().insertAfter(preBlock->getIterator(), loopHeader);
    parentRegion.getBlocks().insertAfter(loopHeader->getIterator(), loopBody);
    parentRegion.getBlocks().insertAfter(loopBody->getIterator(), loopLatch);

    builder.setInsertionPointToEnd(preBlock);
    Value pi = LLVM::AllocaOp::create(builder, loc, ptrT, iterTy, one64);
    LLVM::StoreOp::create(builder, loc, lbCore, pi);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);

    builder.setInsertionPointToEnd(loopHeader);
    Value curI = LLVM::LoadOp::create(builder, loc, iterTy, pi);
    Value cond = LLVM::ICmpOp::create(builder, loc,
      LLVM::ICmpPredicate::slt, curI, ubCore);  // exclusive upper bound
    LLVM::CondBrOp::create(builder, loc, cond,
      loopBody, mlir::ValueRange{}, afterBlock, mlir::ValueRange{});

    // Move the loop nest body into loopBody.
    // The nest region may have multiple blocks (from inner loops).
    // Move all blocks: first block's ops go into loopBody,
    // remaining blocks are spliced in between loopBody and loopLatch.
    auto &nestRegion = loopNest.getRegion();
    auto &nestFirst = nestRegion.front();
    // Replace outer IV with curI.
    nestFirst.getArgument(0).replaceAllUsesWith(curI);

    // Erase omp.yield/omp.terminator from the last block.
    auto &nestLast = nestRegion.back();
    for (auto &innerOp : llvm::make_early_inc_range(nestLast.getOperations()))
      if (innerOp.getName().getStringRef() == "omp.yield" ||
          innerOp.getName().getStringRef() == "omp.terminator")
        innerOp.erase();

    if (nestRegion.hasOneBlock()) {
      // Single block: move all ops into loopBody.
      builder.setInsertionPointToEnd(loopBody);
      SmallVector<Operation *> opsToMove;
      for (auto &innerOp : nestFirst.getOperations())
        opsToMove.push_back(&innerOp);
      for (auto *innerOp : opsToMove)
        innerOp->moveBefore(loopBody, loopBody->end());
      builder.setInsertionPointToEnd(loopBody);
      LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopLatch);
    } else {
      // Multiple blocks (inner loops from CIR): splice ALL blocks before loopLatch
      // first, then move first block's ops into loopBody.
      // This preserves branch targets since blocks are already in the region.
      SmallVector<Block *> blocksToMove;
      for (auto &blk : nestRegion)
        if (&blk != &nestFirst) blocksToMove.push_back(&blk);
      for (auto *blk : blocksToMove)
        blk->moveBefore(loopLatch);

      // Now move nestFirst's ops into loopBody (branch targets are now valid).
      builder.setInsertionPointToEnd(loopBody);
      SmallVector<Operation *> firstOps;
      for (auto &innerOp : nestFirst.getOperations())
        firstOps.push_back(&innerOp);
      for (auto *innerOp : firstOps)
        innerOp->moveBefore(loopBody, loopBody->end());

      // Add branch from last moved block to loopLatch.
      Block *lastBlock = blocksToMove.back();
      builder.setInsertionPointToEnd(lastBlock);
      LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopLatch);
}

    builder.setInsertionPointToEnd(loopLatch);
    Value nextI = LLVM::AddOp::create(builder, loc, curI, step);
    LLVM::StoreOp::create(builder, loc, nextI, pi);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);

    // After loop: emit post-loop calls from DSL (barriers etc).
    builder.setInsertionPointToStart(afterBlock);
    for (auto &action : plan.invoke) {
      if (auto *ca = std::get_if<dsl::PlanCall>(&action)) {
        if (ca->callee == "core_bounds") continue;
        if (ca->callee.find("body") != std::string::npos) continue;
        SmallVector<Value> args;
        SmallVector<Type>  types;
        for (auto &av : ca->args) {
          Value v = resolveCallArg(av);
          args.push_back(v); types.push_back(v.getType());
        }
        auto decl = getOrInsertDecl(module, ca->callee, types, builder);
        func::CallOp::create(builder, loc, decl, args);
      }
    }

  } else if (isGOMP) {
    // -------------------------------------------------------------------------
    // libgomp wsloop: GOMP_loop_static_start / GOMP_loop_static_next pattern
    //
    // GOMP_loop_static_start(long lb, long ub, long step, long chunk,
    //                         long *istart, long *iend) -> bool
    // Pattern:
    //   alloca istart, iend (long = i64)
    //   if (GOMP_loop_static_start(lb, ub, step, chunk, &istart, &iend)):
    //     do {
    //       for i = istart; i < iend; i += step: body(i)
    //     } while (GOMP_loop_static_next(&istart, &iend))
    //   GOMP_loop_end() [or GOMP_loop_end_nowait()]
    // -------------------------------------------------------------------------
    Type longTy = IntegerType::get(ctx, 64);
    Value one64long = LLVM::ConstantOp::create(builder, loc, longTy,
      IntegerAttr::get(longTy, 1));

    // Sign-extend bounds to i64 (GOMP uses long).
    Value lbLong   = LLVM::SExtOp::create(builder, loc, longTy, lb);
    Value ubLong   = LLVM::SExtOp::create(builder, loc, longTy, ub);
    Value stepLong = LLVM::SExtOp::create(builder, loc, longTy, step);
    Value chunkLong = LLVM::ConstantOp::create(builder, loc, longTy,
      IntegerAttr::get(longTy, 1));

    // Allocate istart and iend.
    Value pistart = LLVM::AllocaOp::create(builder, loc, ptrT, longTy, one64);
    Value piend   = LLVM::AllocaOp::create(builder, loc, ptrT, longTy, one64);

    // Call GOMP_loop_static_start.
    SmallVector<Type> startArgTypes = {longTy, longTy, longTy, longTy,
                                        ptrT,   ptrT};
    SmallVector<Value> startArgs    = {lbLong, ubLong, stepLong, chunkLong,
                                        pistart, piend};
    // Returns i1 (bool).
    auto startDecl = getOrInsertDeclWithReturn(module,
      "GOMP_loop_static_start", startArgTypes, IntegerType::get(ctx, 1),
      builder);
    Value hasWork = func::CallOp::create(builder, loc, startDecl,
      startArgs).getResult(0);

    // Split current block: everything after this point goes to afterAll.
    // preBlock → (conditional on hasWork) → chunkLoop or afterAll
    Block *preBlock    = builder.getInsertionBlock();
    Block *afterAll    = preBlock->splitBlock(builder.getInsertionPoint());
    Block *chunkLoop   = new Block();
    Block *innerHeader = new Block();
    Block *innerBody   = new Block();
    Block *innerLatch  = new Block();
    Block *nextChunk   = new Block();

    auto &pr = *preBlock->getParent();
    pr.getBlocks().insertAfter(preBlock->getIterator(),     chunkLoop);
    pr.getBlocks().insertAfter(chunkLoop->getIterator(),    innerHeader);
    pr.getBlocks().insertAfter(innerHeader->getIterator(),  innerBody);
    pr.getBlocks().insertAfter(innerBody->getIterator(),    innerLatch);
    pr.getBlocks().insertAfter(innerLatch->getIterator(),   nextChunk);
    // afterAll is already inserted (the split-off tail)

    // Terminate preBlock: branch to chunkLoop if hasWork, else to afterAll.
    builder.setInsertionPointToEnd(preBlock);
    LLVM::CondBrOp::create(builder, loc, hasWork,
      chunkLoop, mlir::ValueRange{},
      afterAll, mlir::ValueRange{});

    // chunkLoop: load istart/iend, init inner loop.
    builder.setInsertionPointToEnd(chunkLoop);
    Value curStart = LLVM::LoadOp::create(builder, loc, longTy, pistart);
    Value curEnd   = LLVM::LoadOp::create(builder, loc, longTy, piend);
    // Allocate inner induction var.
    Value piInner = LLVM::AllocaOp::create(builder, loc, ptrT, longTy, one64long);
    LLVM::StoreOp::create(builder, loc, curStart, piInner);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, innerHeader);

    // innerHeader: check i < curEnd.
    builder.setInsertionPointToEnd(innerHeader);
    Value curI64 = LLVM::LoadOp::create(builder, loc, longTy, piInner);
    Value innerCond = LLVM::ICmpOp::create(builder, loc,
      LLVM::ICmpPredicate::slt, curI64, curEnd);
    LLVM::CondBrOp::create(builder, loc, innerCond,
      innerBody, mlir::ValueRange{},
      nextChunk, mlir::ValueRange{});

    // innerBody: truncate i64 to iterTy, run loop body.
    builder.setInsertionPointToEnd(innerBody);
    Value curI = LLVM::TruncOp::create(builder, loc, iterTy, curI64);
    auto &gompNestRegion = loopNest.getRegion();
    auto &gompNestFirst  = gompNestRegion.front();
    gompNestFirst.getArgument(0).replaceAllUsesWith(curI);
    // Erase omp.yield/omp.terminator from last block.
    for (auto &innerOp : llvm::make_early_inc_range(gompNestRegion.back().getOperations()))
      if (innerOp.getName().getStringRef() == "omp.yield" ||
          innerOp.getName().getStringRef() == "omp.terminator")
        innerOp.erase();
    if (gompNestRegion.hasOneBlock()) {
      builder.setInsertionPointToEnd(innerBody);
      SmallVector<Operation *> opsToMove;
      for (auto &innerOp : gompNestFirst.getOperations())
        opsToMove.push_back(&innerOp);
      for (auto *innerOp : opsToMove)
        innerOp->moveBefore(innerBody, innerBody->end());
      builder.setInsertionPointToEnd(innerBody);
      LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, innerLatch);
    } else {
      // Multi-block: splice remaining blocks before innerLatch first.
      SmallVector<Block *> extraBlocks;
      for (auto &blk : gompNestRegion)
        if (&blk != &gompNestFirst) extraBlocks.push_back(&blk);
      for (auto *blk : extraBlocks)
        blk->moveBefore(innerLatch);
      // Move first block's ops into innerBody.
      builder.setInsertionPointToEnd(innerBody);
      SmallVector<Operation *> firstOps;
      for (auto &innerOp : gompNestFirst.getOperations())
        firstOps.push_back(&innerOp);
      for (auto *innerOp : firstOps)
        innerOp->moveBefore(innerBody, innerBody->end());
      // Last extra block branches to innerLatch.
      builder.setInsertionPointToEnd(extraBlocks.back());
      LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, innerLatch);
    }

    // innerLatch: i += step.
    builder.setInsertionPointToEnd(innerLatch);
    Value nextI64 = LLVM::AddOp::create(builder, loc, curI64, stepLong);
    LLVM::StoreOp::create(builder, loc, nextI64, piInner);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, innerHeader);

    // nextChunk: call GOMP_loop_static_next, branch back or to afterAll.
    builder.setInsertionPointToEnd(nextChunk);
    auto nextDecl = getOrInsertDeclWithReturn(module,
      "GOMP_loop_static_next", {ptrT, ptrT},
      IntegerType::get(ctx, 1), builder);
    Value hasMore = func::CallOp::create(builder, loc, nextDecl,
      ValueRange{pistart, piend}).getResult(0);
    LLVM::CondBrOp::create(builder, loc, hasMore,
      chunkLoop, mlir::ValueRange{},
      afterAll, mlir::ValueRange{});

    // afterAll: emit post-loop calls (GOMP_loop_end, GOMP_barrier).
    builder.setInsertionPointToStart(afterAll);
    // Collect post calls from plan.
    for (auto &action : plan.invoke) {
      if (auto *ca = std::get_if<dsl::PlanCall>(&action)) {
        if (ca->callee.find("body") != std::string::npos) continue;
        if (ca->callee.find("start") != std::string::npos) continue;
        if (ca->callee.find("next") != std::string::npos) continue;
        SmallVector<Value> args;
        SmallVector<Type>  types;
        for (auto &av : ca->args) {
          Value v = resolveCallArg(av);
          args.push_back(v); types.push_back(v.getType());
        }
        auto decl = getOrInsertDecl(module, ca->callee, types, builder);
        func::CallOp::create(builder, loc, decl, args);
      }
    }

  } else {
    // -------------------------------------------------------------------------
    // iomp wsloop: __kmpc_for_static_init_4 / explicit loop pattern
    // -------------------------------------------------------------------------

    // Separate invoke actions into pre-loop (init), body (skip), post-loop.
    SmallVector<const dsl::PlanCall *> preCalls, postCalls;
    for (auto &action : plan.invoke) {
      if (auto *ca = std::get_if<dsl::PlanCall>(&action)) {
        bool isBody = ca->callee.find("body") != std::string::npos;
        bool isInit = ca->callee.find("init") != std::string::npos;
        if (isBody) continue;
        if (isInit) preCalls.push_back(ca);
        else        postCalls.push_back(ca);
      }
    }

    // Emit pre-loop calls.
    for (auto *ca : preCalls) {
      SmallVector<Value> args;
      SmallVector<Type>  types;
      for (auto &av : ca->args) {
        Value v = resolveCallArg(av);
        args.push_back(v); types.push_back(v.getType());
      }
      auto decl = getOrInsertDecl(module, ca->callee, types, builder);
      func::CallOp::create(builder, loc, decl, args);
    }

    // Load adjusted bounds (written by __kmpc_for_static_init_4).
    // plb/pub contain this thread's iteration range (inclusive).
    // pstride contains the runtime stride (num_threads for static) — NOT used
    // for loop increment. We increment by the original loop step instead.
    Value adjLb   = LLVM::LoadOp::create(builder, loc, iterTy, plb);
    Value adjUb   = LLVM::LoadOp::create(builder, loc, iterTy, pub);

    // Build explicit loop blocks.
    Block *preBlock   = builder.getInsertionBlock();
    Block *afterBlock = preBlock->splitBlock(builder.getInsertionPoint());
    Block *loopHeader = new Block();
    Block *loopBody   = new Block();
    Block *loopLatch  = new Block();

    auto &parentRegion = *preBlock->getParent();
    parentRegion.getBlocks().insertAfter(preBlock->getIterator(), loopHeader);
    parentRegion.getBlocks().insertAfter(loopHeader->getIterator(), loopBody);
    parentRegion.getBlocks().insertAfter(loopBody->getIterator(), loopLatch);

    builder.setInsertionPointToEnd(preBlock);
    Value pi = LLVM::AllocaOp::create(builder, loc, ptrT, iterTy, one64);
    LLVM::StoreOp::create(builder, loc, adjLb, pi);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);

    builder.setInsertionPointToEnd(loopHeader);
    Value curI = LLVM::LoadOp::create(builder, loc, iterTy, pi);
    Value cond = LLVM::ICmpOp::create(builder, loc,
      LLVM::ICmpPredicate::sle, curI, adjUb);
    LLVM::CondBrOp::create(builder, loc, cond,
      loopBody, mlir::ValueRange{}, afterBlock, mlir::ValueRange{});

    // Move the loop nest body into loopBody.
    auto &ioNestRegion = loopNest.getRegion();
    auto &ioNestFirst  = ioNestRegion.front();
    ioNestFirst.getArgument(0).replaceAllUsesWith(curI);
    // Erase omp.yield/omp.terminator from last block.
    for (auto &innerOp : llvm::make_early_inc_range(ioNestRegion.back().getOperations()))
      if (innerOp.getName().getStringRef() == "omp.yield" ||
          innerOp.getName().getStringRef() == "omp.terminator")
        innerOp.erase();
    if (ioNestRegion.hasOneBlock()) {
      builder.setInsertionPointToEnd(loopBody);
      SmallVector<Operation *> opsToMove;
      for (auto &innerOp : ioNestFirst.getOperations())
        opsToMove.push_back(&innerOp);
      for (auto *innerOp : opsToMove)
        innerOp->moveBefore(loopBody, loopBody->end());
      builder.setInsertionPointToEnd(loopBody);
      LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopLatch);
    } else {
      SmallVector<Block *> extraBlocks;
      for (auto &blk : ioNestRegion)
        if (&blk != &ioNestFirst) extraBlocks.push_back(&blk);
      for (auto *blk : extraBlocks)
        blk->moveBefore(loopLatch);
      builder.setInsertionPointToEnd(loopBody);
      SmallVector<Operation *> firstOps;
      for (auto &innerOp : ioNestFirst.getOperations())
        firstOps.push_back(&innerOp);
      for (auto *innerOp : firstOps)
        innerOp->moveBefore(loopBody, loopBody->end());
      builder.setInsertionPointToEnd(extraBlocks.back());
      LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopLatch);
    }

    builder.setInsertionPointToEnd(loopLatch);
    // Increment by original loop step, not by pstride (which is the runtime stride).
    Value nextI = LLVM::AddOp::create(builder, loc, curI, step);
    LLVM::StoreOp::create(builder, loc, nextI, pi);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);

    // Emit post-loop calls.
    builder.setInsertionPointToStart(afterBlock);
    for (auto *ca : postCalls) {
      SmallVector<Value> args;
      SmallVector<Type>  types;
      for (auto &av : ca->args) {
        Value v = resolveCallArg(av);
        args.push_back(v); types.push_back(v.getType());
      }
      auto decl = getOrInsertDecl(module, ca->callee, types, builder);
      func::CallOp::create(builder, loc, decl, args);
    }
  } // end iomp path

  wsOp.erase();
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct OmpOutliningPass
    : public PassWrapper<OmpOutliningPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OmpOutliningPass)

  std::string dslFile;
  std::string runtimeName;

  OmpOutliningPass(std::string dsl, std::string rt)
      : dslFile(std::move(dsl)), runtimeName(std::move(rt)) {}
  OmpOutliningPass(const OmpOutliningPass &) = default;

  llvm::StringRef getArgument()    const override { return "omp-outline"; }
  llvm::StringRef getDescription() const override {
    return "Outline parallel regions and lower wsloop using the DSL file";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, LLVM::LLVMDialect,
                    arith::ArithDialect, omp::OpenMPDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto buf = llvm::MemoryBuffer::getFile(dslFile);
    if (!buf) {
      module.emitError("omp-outline: cannot open DSL file '") << dslFile << "'";
      return signalPassFailure();
    }
    auto program = dsl::parse((*buf)->getBuffer());
    if (!program) {
      module.emitError("omp-outline: DSL parse error: ")
        << llvm::toString(program.takeError());
      return signalPassFailure();
    }
    dsl::Evaluator evaluator(*program);

    // Step 1: outline parallel constructs.
    SmallVector<ConstructOp> constructs;
    module.walk([&](ConstructOp op) {
      if (!op.getBody().empty()) constructs.push_back(op);
    });
    int counter = 0;
    for (auto op : constructs)
      outlineConstruct(op, module, counter, runtimeName);

    // Step 2: lower omp.wsloop ops using the DSL evaluator.
    SmallVector<omp::WsloopOp> wsloops;
    module.walk([&](omp::WsloopOp op) { wsloops.push_back(op); });

    for (auto wsOp : wsloops) {
      if (!wsOp->getBlock()) continue;

      llvm::StringMap<dsl::Value> ctx;
      ctx["ident"]      = dsl::makeStr("%ident");
      ctx["global_tid"] = dsl::makeStr("%tid");
      ctx["lower"]      = dsl::makeStr("%lb");
      ctx["upper"]      = dsl::makeStr("%ub");
      ctx["step"]       = dsl::makeStr("%step");
      ctx["last"]       = dsl::makeStr("last");
      ctx["chunk"]      = dsl::makeInt(1);
      ctx["nowait"]     = dsl::makeBool(wsOp.getNowait());
      ctx["body"]       = dsl::makeStr("body");
      ctx["schedule"]   = dsl::makeStr("static");
      ctx["stride"]     = dsl::makeStr("%stride");  // output ptr for runtime stride
      ctx["stride"]     = dsl::makeStr("%stride");  // output ptr for runtime stride
      if (wsOp.getScheduleKind()) {
        auto sk = omp::stringifyClauseScheduleKind(*wsOp.getScheduleKind());
        ctx["schedule"] = dsl::makeStr(sk.str());
      }

      auto plan = evaluator.buildPlan(runtimeName, "wsloop", ctx);
      if (!plan) {
        wsOp.emitError("omp-outline: wsloop DSL evaluation failed: ")
          << llvm::toString(plan.takeError());
        signalPassFailure();
        return;
      }

      lowerWsloop(wsOp, module, *plan);
    }
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::createOmpOutliningPass(std::string dslFile, std::string runtime) {
  return std::make_unique<OmpOutliningPass>(
      std::move(dslFile), std::move(runtime));
}

void mlir::registerOmpOutliningPass() {
  // Registration handled in mlir-opt-omp.cpp via factory lambda.
}
