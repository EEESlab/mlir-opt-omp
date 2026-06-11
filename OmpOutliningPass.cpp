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
//    - Emit plan.pre (runtime init call OR `emit thread_bounds` → DIVMOD),
//      then an explicit loop, then plan.post

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
#include "mlir/Interfaces/DataLayoutInterfaces.h"
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
  auto allocaOp = val.getDefiningOp<LLVM::AllocaOp>();
  if (!allocaOp) return false;
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

// Returns true if this capture is a scalar alloca whose value should be packed
// by value into the capture struct (instead of storing the alloca pointer).
// Criteria: the captured value is an AllocaOp result whose element type is a
// scalar non-pointer LLVM-compatible type (i.e. integer or float), AND the
// first use of the alloca inside the region is a load (shared-read, not a
// private-write like a loop IV).  These captures correspond to variables like
// alpha and beta in GEMM: their value is read-only inside the parallel region,
// so we can capture the scalar value itself and avoid the extra pointer
// dereference that would otherwise appear on every use inside the inner loop.
static bool isScalarAllocaCapture(Value val, Region &region) {
  auto allocaOp = val.getDefiningOp<LLVM::AllocaOp>();
  if (!allocaOp) return false;
  Type elemTy = allocaOp.getElemType();
  // Must be a scalar non-pointer LLVM type (integer or float).
  if (!LLVM::isCompatibleType(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMPointerType>(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMArrayType>(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMStructType>(elemTy)) return false;
  // The first use in the region must be a load (addr operand), not a store.
  // If it is a store this is a private IV — handled by isPrivateCapture.
  bool firstIsLoad = false, found = false;
  region.walk([&](Operation *op) {
    if (found) return;
    for (auto &use : op->getOpOperands()) {
      if (use.get() == val) {
        found = true;
        if (auto ld = llvm::dyn_cast<LLVM::LoadOp>(op))
          firstIsLoad = true;
        break;
      }
    }
  });
  return firstIsLoad;
}

// Returns true if this capture is a ptr-typed alloca whose stored value
// (a pointer to e.g. an array) should be packed directly into the capture
// struct, eliminating one extra dereference inside the outlined function.
// Criteria: alloca with element type ptr, and the first use in the region
// is a load (reading the stored pointer, not writing it — i.e. not a private IV).
static bool isPtrAllocaCapture(Value val, Region &region) {
  auto allocaOp = val.getDefiningOp<LLVM::AllocaOp>();
  if (!allocaOp) return false;
  Type elemTy = allocaOp.getElemType();
  if (!llvm::isa<LLVM::LLVMPointerType>(elemTy)) return false;
  bool firstIsLoad = false, found = false;
  region.walk([&](Operation *op) {
    if (found) return;
    for (auto &use : op->getOpOperands()) {
      if (use.get() == val) {
        found = true;
        if (llvm::isa<LLVM::LoadOp>(op)) firstIsLoad = true;
        break;
      }
    }
  });
  return firstIsLoad;
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
                              const dsl::LoweringPlan &barrierPlan) {
  Region &body = op.getBody();
  if (body.empty()) return;

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();

  std::string outlineSig   = getPropStr(op, "outline_signature");
  std::string captureStrat = getPropStr(op, "capture_strategy");

  bool isMicrotask = outlineSig.find("microtask")  != std::string::npos;
  bool isClosure   = outlineSig.find("closure")    != std::string::npos;
  bool isTaskEntry = outlineSig.find("task_entry") != std::string::npos;
  bool isPacked    = captureStrat == "packed";

  // Collect all values used inside the region but defined outside.
  SmallVector<Value> captures = collectCaptures(body);

  // Compute private captures before takeBody (body region is consumed after).
  llvm::SetVector<Value> privateCaptures;
  for (auto cap : captures)
    if (isPrivateCapture(cap, body))
      privateCaptures.insert(cap);

  // Compute scalar-alloca captures: allocas holding a scalar (int/float) whose
  // first use in the region is a load.  For the packed/closure strategy these
  // will be stored by value in the capture struct so the outlined function
  // receives the scalar directly — eliminating the double indirection (struct
  // field → alloca pointer → scalar) that would otherwise appear on every use
  // inside the innermost loop and block LLVM's LICM from hoisting those loads.
  llvm::SetVector<Value> scalarAllocaCaptures;
  for (auto cap : captures)
    if (!privateCaptures.contains(cap) && isScalarAllocaCapture(cap, body))
      scalarAllocaCaptures.insert(cap);

  // Compute ptr-alloca captures: allocas holding a pointer (e.g. array ptr)
  // whose first use in the region is a load.  Pack the pointer VALUE directly
  // into the struct field instead of the alloca address, saving one dereference
  // per array access inside the outlined function.
  llvm::SetVector<Value> ptrAllocaCaptures;
  for (auto cap : captures)
    if (!privateCaptures.contains(cap) && !scalarAllocaCaptures.contains(cap) &&
        isPtrAllocaCapture(cap, body))
      ptrAllocaCaptures.insert(cap);

  std::string fnName =
    "outlined_" + op.getConstructName().str() + "_" + std::to_string(counter++);

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
  // Use nested visibility: the outlined function is referenced via
  // func.constant which --remove-dead-values may not track after lowering.
  // Nested visibility prevents the symbol from being eliminated by DCE.
  outlinedFn.setVisibility(SymbolTable::Visibility::Nested);
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
    // Insert the leading args and determine `dataPtr`, the pointer to the
    // capture struct that the unpack loop reads from.
    //   closure          : (ptr data)                  -> dataPtr = arg0
    //   microtask+packed : (ptr gtid, ptr btid, ptr d) -> dataPtr = arg2
    //   task_entry (iomp): (i32 gtid, ptr task)        -> dataPtr = load(task)
    //                      (captures live behind task->shareds, field 0)
    Value dataPtr;
    BlockArgument taskArg; // iomp task entry only
    if (isTaskEntry) {
      entry.insertArgument(0u, ptrTy(ctx), loc); // task (becomes arg 1)
      entry.insertArgument(0u, i32Ty(ctx), loc); // gtid (arg 0)
      taskArg = entry.getArgument(1);
      // dataPtr = load(task) is materialised at the start of the unpack
      // prologue below (so it precedes the GEPs that use it).
    } else {
      if (isMicrotask) {
        entry.insertArgument(0u, ptrTy(ctx), loc); // btid
        entry.insertArgument(0u, ptrTy(ctx), loc); // gtid
      }
      // The entry block may already have privatizer args — insert before them.
      unsigned dataPtrIdx = isMicrotask ? 2u : 0u;
      entry.insertArgument(dataPtrIdx, ptrTy(ctx), loc);
      dataPtr = entry.getArgument(dataPtrIdx);
    }

    // Build the capture struct type: { cap_0_type, cap_1_type, ... }
    // Scalar-alloca captures use their element type as the field type (value
    // semantics); all other captures use the captured value's own type (ptr).
    SmallVector<Type> fieldTypes;
    for (auto cap : captures) {
      if (scalarAllocaCaptures.contains(cap))
        fieldTypes.push_back(
            cap.getDefiningOp<LLVM::AllocaOp>().getElemType());
      else
        fieldTypes.push_back(cap.getType());
    }
    auto structTy = LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);

    // In the function prolog, unpack each capture from the struct.
    if (!captures.empty()) {
      OpBuilder prologue(&entry, entry.begin());
      Value one64 = LLVM::ConstantOp::create(prologue, loc,
        IntegerType::get(ctx, 64),
        IntegerAttr::get(IntegerType::get(ctx, 64), 1));
      // For the iomp task entry, the capture struct sits behind task->shareds
      // (field 0 of kmp_task_t); load it here before any GEP uses it.
      if (isTaskEntry)
        dataPtr = LLVM::LoadOp::create(prologue, loc, ptrTy(ctx), taskArg);
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
        } else if (scalarAllocaCaptures.contains(captures[i])) {
          // Scalar-alloca capture packed by value: the struct field already
          // holds the scalar (e.g. double), not a pointer to it.  Load the
          // scalar directly from the struct — one load, no further dereference.
          // All uses of the original alloca (previously a ptr) inside the body
          // were load/store ops on that alloca; replace the alloca ptr with a
          // fresh local alloca that is initialised with the unpacked scalar so
          // existing load/store patterns in the body remain valid without any
          // further IR surgery.
          Type elemTy = captures[i].getDefiningOp<LLVM::AllocaOp>().getElemType();
          Value gep = LLVM::GEPOp::create(prologue, loc, ptrTy(ctx), structTy,
            dataPtr, ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
          Value scalarVal = LLVM::LoadOp::create(prologue, loc, elemTy, gep);
          // Place the scalar into a fresh alloca so that any load/store uses
          // of the original alloca inside the body continue to work.  LLVM's
          // mem2reg/SROA will promote this single-store alloca to a register,
          // making the value a true SSA constant visible to LICM.
          Value privAlloca = LLVM::AllocaOp::create(prologue, loc,
            ptrTy(ctx), elemTy, one64);
          LLVM::StoreOp::create(prologue, loc, scalarVal, privAlloca);
          loadedCaptures.push_back(privAlloca);
          replaceUsesInRegion(outlinedFn.getBody(), captures[i], privAlloca);
        } else if (ptrAllocaCaptures.contains(captures[i])) {
          // Ptr-alloca capture: the struct field holds the pointer value
          // directly.  Unpack it into a fresh alloca so that existing
          // load/store-of-alloca patterns in the body continue to work.
          Value gep = LLVM::GEPOp::create(prologue, loc, ptrTy(ctx), structTy,
            dataPtr, ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
          Value ptrVal = LLVM::LoadOp::create(prologue, loc, ptrTy(ctx), gep);
          Value privAlloca = LLVM::AllocaOp::create(prologue, loc,
            ptrTy(ctx), ptrTy(ctx), one64);
          LLVM::StoreOp::create(prologue, loc, ptrVal, privAlloca);
          loadedCaptures.push_back(privAlloca);
          replaceUsesInRegion(outlinedFn.getBody(), captures[i], privAlloca);
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

  // Update function type to match actual entry block args (preserving the
  // i32 result for the iomp task entry).
  SmallVector<Type> finalArgTypes;
  for (auto arg : entry.getArguments())
    finalArgTypes.push_back(arg.getType());
  outlinedFn.setFunctionType(FunctionType::get(ctx, finalArgTypes,
    isTaskEntry ? TypeRange{i32Ty(ctx)} : TypeRange{}));

  // Replace omp.terminator with func.return (direct blocks only).
  SmallVector<Operation *> terminators;
  for (auto &block : outlinedFn.getBody())
    for (auto &innerOp : block)
      if (innerOp.getName().getStringRef() == "omp.terminator")
        terminators.push_back(&innerOp);
  for (auto *termOp : terminators) {
    OpBuilder tb(termOp);
    if (isTaskEntry) {
      // iomp task entry returns kmp_int32 (0).
      Value zero = LLVM::ConstantOp::create(tb, termOp->getLoc(), i32Ty(ctx),
        IntegerAttr::get(i32Ty(ctx), 0));
      func::ReturnOp::create(tb, termOp->getLoc(), ValueRange{zero});
    } else {
      func::ReturnOp::create(tb, termOp->getLoc());
    }
    termOp->erase();
  }

  // Lower omp.barrier inside the outlined function — DSL-driven.
  // The barrier plan (one per runtime) is built once in runOnOperation;
  // here we only resolve the symbolic args (%ident, %tid) at each call site.
  SmallVector<Operation *> barriers;
  for (auto &block : outlinedFn.getBody())
    for (auto &innerOp : block)
      if (innerOp.getName().getStringRef() == "omp.barrier")
        barriers.push_back(&innerOp);
  for (auto *barrierOp : barriers) {
    OpBuilder bb(barrierOp);
    Location bloc = barrierOp->getLoc();

    auto resolveArg = [&](const dsl::Value &v) -> Value {
      if (auto *sv = std::get_if<dsl::StrVal>(&v)) {
        llvm::StringRef s = sv->value;
        if (s == "%ident" || s == "ident")
          return getIdentAddr(module, bb, bloc, ctx);
        if (s == "%tid" || s == "global_tid") {
          // Microtask convention: first arg is ptr to i32 gtid.
          auto &fnEntry = outlinedFn.getBody().front();
          if (fnEntry.getNumArguments() >= 2)
            return LLVM::LoadOp::create(bb, bloc, i32Ty(ctx),
                                         fnEntry.getArgument(0));
          return LLVM::UndefOp::create(bb, bloc, i32Ty(ctx));
        }
      }
      if (auto *iv = std::get_if<dsl::IntVal>(&v))
        return arith::ConstantOp::create(bb, bloc, i32Ty(ctx),
          IntegerAttr::get(i32Ty(ctx), iv->value));
      return LLVM::UndefOp::create(bb, bloc, ptrTy(ctx));
    };

    for (auto &action : barrierPlan.invoke) {
      if (auto *ca = std::get_if<dsl::PlanCall>(&action)) {
        SmallVector<Value> args;
        SmallVector<Type>  types;
        for (auto &av : ca->args) {
          Value v = resolveArg(av);
          args.push_back(v); types.push_back(v.getType());
        }
        auto decl = getOrInsertDecl(module, ca->callee, types, bb);
        func::CallOp::create(bb, bloc, decl, args);
      }
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
          else if (s == "num_threads" && op.getNumThreads())
            v = op.getNumThreads();
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

    if (isTaskEntry) {
      // -----------------------------------------------------------------------
      // iomp task: __kmpc_omp_task_alloc -> fill task->shareds -> __kmpc_omp_task
      // -----------------------------------------------------------------------
      // ABI constants — VERIFY against the target's kmp.h / clang codegen:
      //   TASK_FLAGS = 1        (tied task)
      //   KMP_TASK_T_SIZE = 40  (sizeof(kmp_task_t); the runtime places shareds
      //                          right after this header and sets task->shareds
      //                          to point there. Must be >= the runtime's real
      //                          kmp_task_t size, else shareds overlap it.)
      const int64_t TASK_FLAGS      = 1;
      const int64_t KMP_TASK_T_SIZE = 40;

      // Capture struct field types (same logic as the packed path).
      SmallVector<Type> fieldTypes;
      for (auto cap : captures) {
        if (scalarAllocaCaptures.contains(cap))
          fieldTypes.push_back(cap.getDefiningOp<LLVM::AllocaOp>().getElemType());
        else
          fieldTypes.push_back(cap.getType());
      }
      auto structTy = LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);
      auto i64Ty = IntegerType::get(ctx, 64);
      uint64_t sharedsSize =
        DataLayout(module).getTypeSize(structTy).getFixedValue();

      Value gtidVal = gtidAtCallSite
        ? gtidAtCallSite
        : LLVM::UndefOp::create(builder, loc, i32Ty(ctx));

      // Entry function pointer (cast to opaque ptr for the runtime call).
      Value entryPtr = func::ConstantOp::create(builder, loc,
        outlinedFn.getFunctionType(), FlatSymbolRefAttr::get(ctx, fnName));
      Value entryCast = UnrealizedConversionCastOp::create(
        builder, loc, TypeRange{ptrTy(ctx)}, ValueRange{entryPtr}).getResult(0);

      auto i32c = [&](int64_t v) -> Value {
        return arith::ConstantOp::create(builder, loc, i32Ty(ctx),
          IntegerAttr::get(i32Ty(ctx), v));
      };
      auto i64c = [&](int64_t v) -> Value {
        return LLVM::ConstantOp::create(builder, loc, i64Ty,
          IntegerAttr::get(i64Ty, v));
      };

      // task = __kmpc_omp_task_alloc(ident, gtid, flags, sizeof_kmp_task_t,
      //                              sizeof_shareds, entry)
      SmallVector<Type> allocArgTys = {ptrTy(ctx), i32Ty(ctx), i32Ty(ctx),
                                       i64Ty, i64Ty, ptrTy(ctx)};
      auto allocDecl = getOrInsertDeclWithReturn(module,
        "__kmpc_omp_task_alloc", allocArgTys, ptrTy(ctx), builder);
      Value task = func::CallOp::create(builder, loc, allocDecl,
        ValueRange{identVal, gtidVal, i32c(TASK_FLAGS),
                   i64c(KMP_TASK_T_SIZE), i64c((int64_t)sharedsSize),
                   entryCast}).getResult(0);

      // shareds = task->shareds (field 0 of kmp_task_t).
      Value shareds = LLVM::LoadOp::create(builder, loc, ptrTy(ctx), task);

      // Store each capture into the shareds struct.
      for (size_t i = 0; i < captures.size(); i++) {
        Value capVal;
        if (privateCaptures.contains(captures[i])) {
          capVal = LLVM::UndefOp::create(builder, loc, fieldTypes[i]);
        } else if (scalarAllocaCaptures.contains(captures[i])) {
          Type elemTy = captures[i].getDefiningOp<LLVM::AllocaOp>().getElemType();
          capVal = LLVM::LoadOp::create(builder, loc, elemTy, captures[i]);
        } else if (ptrAllocaCaptures.contains(captures[i])) {
          capVal = LLVM::LoadOp::create(builder, loc, ptrTy(ctx), captures[i]);
        } else {
          capVal = captures[i];
        }
        Value gep = LLVM::GEPOp::create(builder, loc, ptrTy(ctx), structTy,
          shareds, ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
        LLVM::StoreOp::create(builder, loc, capVal, gep);
      }

      // __kmpc_omp_task(ident, gtid, task)
      auto submitDecl = getOrInsertDeclWithReturn(module, "__kmpc_omp_task",
        {ptrTy(ctx), i32Ty(ctx), ptrTy(ctx)}, i32Ty(ctx), builder);
      func::CallOp::create(builder, loc, submitDecl,
        ValueRange{identVal, gtidVal, task});

      op.erase();
      return;
    }

    if (isPacked) {
      // -----------------------------------------------------------------------
      // PACKED / CLOSURE: build capture struct on the stack, pass ptr to it
      // -----------------------------------------------------------------------
      // Scalar-alloca captures (e.g. alpha, beta) use their element type as the
      // field type so the outlined function receives the scalar directly.  All
      // other captures retain pointer semantics (field type == cap.getType()).
      SmallVector<Type> fieldTypes;
      for (auto cap : captures) {
        if (scalarAllocaCaptures.contains(cap))
          fieldTypes.push_back(
              cap.getDefiningOp<LLVM::AllocaOp>().getElemType());
        else
          fieldTypes.push_back(cap.getType());
      }

      // Build the capture struct (same for all packed runtimes).  structTy is
      // hoisted here (not scoped to the non-empty case) so env_size / env_align
      // can be resolved from it in the invoke arg loop below.
      auto structTy = LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);
      Value structAlloca;
      if (!captures.empty()) {
        Value one64 = LLVM::ConstantOp::create(builder, loc,
          IntegerType::get(ctx, 64),
          IntegerAttr::get(IntegerType::get(ctx, 64), 1));
        structAlloca = LLVM::AllocaOp::create(builder, loc,
          ptrTy(ctx), structTy, one64);
        for (size_t i = 0; i < captures.size(); i++) {
          Value capVal;
          if (privateCaptures.contains(captures[i])) {
            // Private capture — slot not read in outlined fn; pass undef.
            capVal = LLVM::UndefOp::create(builder, loc, fieldTypes[i]);
          } else if (scalarAllocaCaptures.contains(captures[i])) {
            // Scalar-alloca capture: load the scalar value from the alloca at
            // the parallel call site and store the scalar into the struct field.
            // The outlined function will unpack it with a single load — no
            // second pointer dereference anywhere inside the parallel region.
            Type elemTy = captures[i].getDefiningOp<LLVM::AllocaOp>().getElemType();
            capVal = LLVM::LoadOp::create(builder, loc, elemTy, captures[i]);
          } else if (ptrAllocaCaptures.contains(captures[i])) {
            // Ptr-alloca capture: load the pointer value from the alloca and
            // store the pointer directly into the struct field — eliminates one
            // extra dereference (struct field → alloca → ptr → array).
            capVal = LLVM::LoadOp::create(builder, loc, ptrTy(ctx), captures[i]);
          } else {
            capVal = captures[i];
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
      //   "body"        → fnPtrCast  (the outlined function pointer)
      //   "env_ptr"     → structAlloca (the capture struct pointer)
      //   "env_size"    → sizeof(structTy)  as i64  (GOMP_task arg_size)
      //   "env_align"   → alignof(structTy) as i64  (GOMP_task arg_align)
      //   "null"        → null pointer              (GOMP_task cpyfn / depend)
      //   "num_threads" → op.getNumThreads() (SSA operand from omp.parallel)
      //   bool          → i8 constant (libgomp passes C bool as a byte)
      //   integer       → i32 constant
      //   other str     → undef ptr
      auto i64Ty = IntegerType::get(ctx, 64);
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
            else if (s == "env_size") {
              uint64_t sz =
                DataLayout(module).getTypeSize(structTy).getFixedValue();
              v = LLVM::ConstantOp::create(builder, loc, i64Ty,
                IntegerAttr::get(i64Ty, (int64_t)sz));
            } else if (s == "env_align") {
              uint64_t al = DataLayout(module).getTypeABIAlignment(structTy);
              v = LLVM::ConstantOp::create(builder, loc, i64Ty,
                IntegerAttr::get(i64Ty, (int64_t)al));
            } else if (s == "null") {
              v = LLVM::ZeroOp::create(builder, loc, ptrTy(ctx));
            } else if (s == "num_threads" && op.getNumThreads())
              v = op.getNumThreads();
            else
              v = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
          } else if (auto ba = llvm::dyn_cast<BoolAttr>(argAttr)) {
            // Check BoolAttr before IntegerAttr: in MLIR a BoolAttr *is* an
            // i1 IntegerAttr, so the IntegerAttr branch would also match it.
            auto i8Ty = IntegerType::get(ctx, 8);
            v = LLVM::ConstantOp::create(builder, loc, i8Ty,
              IntegerAttr::get(i8Ty, ba.getValue() ? 1 : 0));
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

    // PMSIS task: the DSL invoke is `call body(env_ptr)`, i.e. the callee is
    // the outlined function itself.  Emit a direct synchronous call — the
    // cluster has no task runtime, so the task runs inline on the encountering
    // core (correct for independent tasks; serialises concurrent ones).
    if (runtimeCallee == "body" || runtimeCallee == "outlined_parallel" ||
        runtimeCallee == "outlined_task") {
      func::CallOp::create(builder, loc, outlinedFn, callArgs);
    } else
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
        llvmDecl = LLVM::LLVMFuncOp::create(builder,
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

  // omp.loop_nest bounds may be inclusive or exclusive depending on how the
  // front-end emitted them (the "inclusive" keyword on the op means the upper
  // bound is the last valid iteration value, not one-past-the-end).
  // Normalise to exclusive here so all downstream trip-count and range
  // arithmetic is uniform: exclusive_ub = inclusive_ub + step.
  if (loopNest.getLoopInclusive())
    ub = LLVM::AddOp::create(builder, loc, ub, step);

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

  // Read DSL properties that drive loop lowering strategy.
  auto getStrProp = [&](llvm::StringRef key) -> std::string {
    auto it = plan.properties.find(key.str());
    if (it == plan.properties.end()) return "";
    if (auto *sv = std::get_if<dsl::StrVal>(&it->second)) return sv->value;
    return "";
  };

  // Emit each PlanCall in a block (e.g. plan.pre / plan.post) at the
  // current insertion point of `b`. Symbolic args are resolved via
  // resolveCallArg above.
  auto emitPlanCalls = [&](const std::vector<dsl::PlanAction> &actions,
                           OpBuilder &b) {
    for (auto &action : actions) {
      auto *ca = std::get_if<dsl::PlanCall>(&action);
      if (!ca) continue;
      SmallVector<Value> args;
      SmallVector<Type>  types;
      for (auto &av : ca->args) {
        Value v = resolveCallArg(av);
        args.push_back(v); types.push_back(v.getType());
      }
      auto decl = getOrInsertDecl(module, ca->callee, types, b);
      func::CallOp::create(b, loc, decl, args);
    }
  };

  // Move the loop_nest body into `bodyBlock`, bind the induction variable to
  // `curI`, and append a branch to `latchBlock`. Handles single- and
  // multi-block nests (the latter arising from inner loops in the CIR input).
  // Shared by the single-loop and chunked (dynamic) lowering paths.
  auto moveLoopBody = [&](Block *bodyBlock, Block *latchBlock, Value curI) {
    auto &nestRegion = loopNest.getRegion();
    auto &nestFirst  = nestRegion.front();
    nestFirst.getArgument(0).replaceAllUsesWith(curI);
    for (auto &innerOp :
         llvm::make_early_inc_range(nestRegion.back().getOperations()))
      if (innerOp.getName().getStringRef() == "omp.yield" ||
          innerOp.getName().getStringRef() == "omp.terminator")
        innerOp.erase();

    if (nestRegion.hasOneBlock()) {
      builder.setInsertionPointToEnd(bodyBlock);
      SmallVector<Operation *> opsToMove;
      for (auto &innerOp : nestFirst.getOperations())
        opsToMove.push_back(&innerOp);
      for (auto *innerOp : opsToMove)
        innerOp->moveBefore(bodyBlock, bodyBlock->end());
      builder.setInsertionPointToEnd(bodyBlock);
      LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, latchBlock);
    } else {
      // Multiple blocks (inner loops from CIR): splice ALL blocks before
      // latchBlock first, then move first block's ops into bodyBlock — this
      // preserves branch targets since blocks are already in the region.
      SmallVector<Block *> blocksToMove;
      for (auto &blk : nestRegion)
        if (&blk != &nestFirst) blocksToMove.push_back(&blk);
      for (auto *blk : blocksToMove)
        blk->moveBefore(latchBlock);

      builder.setInsertionPointToEnd(bodyBlock);
      SmallVector<Operation *> firstOps;
      for (auto &innerOp : nestFirst.getOperations())
        firstOps.push_back(&innerOp);
      for (auto *innerOp : firstOps)
        innerOp->moveBefore(bodyBlock, bodyBlock->end());

      builder.setInsertionPointToEnd(blocksToMove.back());
      LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, latchBlock);
    }
  };

  // ---------------------------------------------------------------------------
  // Process plan.pre: emit each PlanCall directly; if `emit thread_bounds`
  // is encountered, materialise per-thread [lbCore, ubCore) inline via the
  // DIVMOD formula (matches GCC's _omp_fn ABI).  The runtime path (e.g. iomp
  // __kmpc_for_static_init_4) has only PlanCalls in pre; the inline path
  // (PMSIS, libgomp) has only `emit thread_bounds`.  Whether `emit
  // thread_bounds` was seen drives the loop bounds choice below.
  // ---------------------------------------------------------------------------
  Value lbCore, ubCore;
  bool haveInlineBounds = false;
  bool haveChunkedLoop  = false;
  bool haveDispatchLoop = false;
  std::string chunkStartFn, chunkNextFn;
  std::string dispatchInitFn, dispatchNextFn;
  for (auto &action : plan.pre) {
    if (auto *ca = std::get_if<dsl::PlanCall>(&action)) {
      SmallVector<Value> args;
      SmallVector<Type>  types;
      for (auto &av : ca->args) {
        Value v = resolveCallArg(av);
        args.push_back(v); types.push_back(v.getType());
      }
      auto decl = getOrInsertDecl(module, ca->callee, types, builder);
      func::CallOp::create(builder, loc, decl, args);
      continue;
    }
    auto *pe = std::get_if<dsl::PlanEmit>(&action);
    if (!pe) continue;
    if (pe->name == "chunked_loop") {
      // libgomp dynamic: start/next chunk acquisition (i64, half-open).
      // The retry-loop CFG is built below; fn names come from properties.
      haveChunkedLoop = true;
      chunkStartFn = getStrProp("chunk_start_function");
      chunkNextFn  = getStrProp("chunk_next_function");
      continue;
    }
    if (pe->name == "dispatch_loop") {
      // iomp dynamic: __kmpc_dispatch_init_4 + while(__kmpc_dispatch_next_4)
      // (i32, inclusive ub). The dispatch CFG is built below.
      haveDispatchLoop = true;
      dispatchInitFn = getStrProp("dispatch_init_function");
      dispatchNextFn = getStrProp("dispatch_next_function");
      continue;
    }
    if (pe->name != "thread_bounds") continue;

    // DIVMOD work-distribution:
    //   trip = ceil((ub - lb) / step)
    //   q    = trip / num_threads  -- base chunk size for every thread
    //   tt   = trip % num_threads  -- first `tt` threads get one extra iteration
    //   thread_start = q * threadId + min(threadId, tt)
    //   thread_end   = thread_start + q + (threadId < tt ? 1 : 0)
    // Tiles [0, trip) with no overlap and no gap.  The helper function names
    // come from the DSL properties thread_id_function / num_thread_function,
    // so this code path serves any runtime that exposes such helpers.
    std::string threadIdFn  = getStrProp("thread_id_function");
    std::string numThreadFn = getStrProp("num_thread_function");
    Value threadId   = emitNoArgI32Call(module, builder, loc, threadIdFn);
    Value numThreads = emitNoArgI32Call(module, builder, loc, numThreadFn);

    Value range    = LLVM::SubOp::create(builder, loc, ub, lb);
    Value rangeS   = LLVM::AddOp::create(builder, loc, range,
                       LLVM::SubOp::create(builder, loc, step, one32));
    Value trip     = LLVM::SDivOp::create(builder, loc, rangeS, step);

    Value q  = LLVM::SDivOp::create(builder, loc, trip, numThreads);
    Value tt = LLVM::SRemOp::create(builder, loc, trip, numThreads);

    Value ltTt       = LLVM::ICmpOp::create(builder, loc,
                         LLVM::ICmpPredicate::slt, threadId, tt);
    Value minThreadIdTt = LLVM::SelectOp::create(builder, loc, ltTt, threadId, tt);

    Value qMulId     = LLVM::MulOp::create(builder, loc, q, threadId);
    Value threadStart = LLVM::AddOp::create(builder, loc, qMulId, minThreadIdTt);

    Value extraIter  = LLVM::SelectOp::create(builder, loc, ltTt, one32, zero32);
    Value threadEnd  = LLVM::AddOp::create(builder, loc,
                         LLVM::AddOp::create(builder, loc, threadStart, q),
                         extraIter);

    // Map back to the original index space.
    // lbCore = lb + thread_start * step  (inclusive start)
    // ubCore = lb + thread_end   * step  (exclusive end — exact, no clamp)
    lbCore = LLVM::AddOp::create(builder, loc, lb,
               LLVM::MulOp::create(builder, loc, threadStart, step));
    ubCore = LLVM::AddOp::create(builder, loc, lb,
               LLVM::MulOp::create(builder, loc, threadEnd, step));
    haveInlineBounds = true;
  }

  // -------------------------------------------------------------------------
  // Chunked (dynamic / guided / runtime) wsloop.
  // -------------------------------------------------------------------------
  // The runtime hands out chunks one at a time:
  //   if (start(lb, ub, step, chunk, &istart, &iend))
  //     do { for (i = istart; i < iend; i += step) body; }
  //     while (next(&istart, &iend));
  //   <post: loop_end / loop_end_nowait>
  // GOMP_loop_*_start / _next use `long` (i64): bounds are sign-extended to i64
  // for the calls and the loaded half-open chunk bounds truncated back to
  // iterTy for the body's induction variable.
  if (haveChunkedLoop) {
    auto longTy = IntegerType::get(ctx, 64);
    Value one64long = LLVM::ConstantOp::create(builder, loc, longTy,
      IntegerAttr::get(longTy, 1));
    // chunk = 1 for now; schedule(dynamic, N) would thread N as an operand
    // (same pattern as num_threads).
    Value chunkLong = LLVM::ConstantOp::create(builder, loc, longTy,
      IntegerAttr::get(longTy, 1));

    Value lbLong   = LLVM::SExtOp::create(builder, loc, longTy, lb);
    Value ubLong   = LLVM::SExtOp::create(builder, loc, longTy, ub);
    Value stepLong = LLVM::SExtOp::create(builder, loc, longTy, step);

    Value pistart = LLVM::AllocaOp::create(builder, loc, ptrT, longTy, one64long);
    Value piend   = LLVM::AllocaOp::create(builder, loc, ptrT, longTy, one64long);

    // hasWork = start(lb, ub, step, chunk, &istart, &iend) -> i1
    SmallVector<Type> startArgTys = {longTy, longTy, longTy, longTy, ptrT, ptrT};
    auto startDecl = getOrInsertDeclWithReturn(module, chunkStartFn,
      startArgTys, IntegerType::get(ctx, 1), builder);
    Value hasWork = func::CallOp::create(builder, loc, startDecl,
      ValueRange{lbLong, ubLong, stepLong, chunkLong,
                 pistart, piend}).getResult(0);

    Block *preBlock    = builder.getInsertionBlock();
    Block *afterAll    = preBlock->splitBlock(builder.getInsertionPoint());
    Block *chunkLoop   = new Block();
    Block *innerHeader = new Block();
    Block *innerBody   = new Block();
    Block *innerLatch  = new Block();
    Block *nextChunk   = new Block();

    auto &pr = *preBlock->getParent();
    pr.getBlocks().insertAfter(preBlock->getIterator(),    chunkLoop);
    pr.getBlocks().insertAfter(chunkLoop->getIterator(),   innerHeader);
    pr.getBlocks().insertAfter(innerHeader->getIterator(), innerBody);
    pr.getBlocks().insertAfter(innerBody->getIterator(),   innerLatch);
    pr.getBlocks().insertAfter(innerLatch->getIterator(),  nextChunk);

    builder.setInsertionPointToEnd(preBlock);
    LLVM::CondBrOp::create(builder, loc, hasWork,
      chunkLoop, mlir::ValueRange{}, afterAll, mlir::ValueRange{});

    // chunkLoop: load this chunk's [istart, iend), init the inner IV.
    builder.setInsertionPointToEnd(chunkLoop);
    Value curStart = LLVM::LoadOp::create(builder, loc, longTy, pistart);
    Value curEnd   = LLVM::LoadOp::create(builder, loc, longTy, piend);
    Value piInner  = LLVM::AllocaOp::create(builder, loc, ptrT, longTy, one64long);
    LLVM::StoreOp::create(builder, loc, curStart, piInner);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, innerHeader);

    // innerHeader: i < iend ? innerBody : nextChunk   (half-open, slt)
    builder.setInsertionPointToEnd(innerHeader);
    Value curI64 = LLVM::LoadOp::create(builder, loc, longTy, piInner);
    Value innerCond = LLVM::ICmpOp::create(builder, loc,
      LLVM::ICmpPredicate::slt, curI64, curEnd);
    LLVM::CondBrOp::create(builder, loc, innerCond,
      innerBody, mlir::ValueRange{}, nextChunk, mlir::ValueRange{});

    // innerBody: truncate the i64 IV to iterTy and run the loop body.
    builder.setInsertionPointToEnd(innerBody);
    Value curI = LLVM::TruncOp::create(builder, loc, iterTy, curI64);
    moveLoopBody(innerBody, innerLatch, curI);

    // innerLatch: i += step
    builder.setInsertionPointToEnd(innerLatch);
    Value nextI64 = LLVM::AddOp::create(builder, loc, curI64, stepLong);
    LLVM::StoreOp::create(builder, loc, nextI64, piInner);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, innerHeader);

    // nextChunk: next(&istart, &iend) ? chunkLoop : afterAll
    builder.setInsertionPointToEnd(nextChunk);
    auto nextDecl = getOrInsertDeclWithReturn(module, chunkNextFn,
      {ptrT, ptrT}, IntegerType::get(ctx, 1), builder);
    Value hasMore = func::CallOp::create(builder, loc, nextDecl,
      ValueRange{pistart, piend}).getResult(0);
    LLVM::CondBrOp::create(builder, loc, hasMore,
      chunkLoop, mlir::ValueRange{}, afterAll, mlir::ValueRange{});

    // afterAll: post calls (loop_end / loop_end_nowait, optional barrier).
    builder.setInsertionPointToStart(afterAll);
    emitPlanCalls(plan.post, builder);

    wsOp.erase();
    return;
  }

  // -------------------------------------------------------------------------
  // iomp dynamic wsloop: __kmpc_dispatch_init_4 + while(__kmpc_dispatch_next_4).
  // -------------------------------------------------------------------------
  //   dispatch_init(ident, gtid, sched, lb, ub_incl, step, chunk)
  //   while (dispatch_next(ident, gtid, &last, &lb, &ub, &stride))
  //     for (i = lb; i <= ub; i += step) body;   // iomp chunk bounds inclusive
  //   <post: barrier unless nowait>
  // i32 bounds throughout (no SExt/Trunc). dispatch_next returns kmp_int32
  // (0 = no more work). Reuses plb/pub/pstride/plast as the runtime out params.
  if (haveDispatchLoop) {
    // ABI: kmp_sch_dynamic_chunked = 35 — VERIFY against the target's kmp.h.
    const int64_t KMP_SCH_DYNAMIC_CHUNKED = 35;
    Value sched = LLVM::ConstantOp::create(builder, loc, iterTy,
      IntegerAttr::get(iterTy, KMP_SCH_DYNAMIC_CHUNKED));
    Value chunk = LLVM::ConstantOp::create(builder, loc, iterTy,
      IntegerAttr::get(iterTy, 1)); // chunk = 1 (first cut)

    // dispatch_init(ident, gtid, sched, lb, ub_incl, step, chunk) -> void
    SmallVector<Type> initArgTys = {ptrT, iterTy, iterTy, iterTy, iterTy,
                                    iterTy, iterTy};
    auto initDecl = getOrInsertDecl(module, dispatchInitFn, initArgTys, builder);
    func::CallOp::create(builder, loc, initDecl,
      ValueRange{identAddr, gtidVal, sched, lb, ubInclusive, step, chunk});

    Block *preBlock       = builder.getInsertionBlock();
    Block *afterAll       = preBlock->splitBlock(builder.getInsertionPoint());
    Block *dispatchHeader = new Block();
    Block *chunkBody      = new Block();
    Block *innerHeader    = new Block();
    Block *innerBody      = new Block();
    Block *innerLatch     = new Block();

    auto &pr = *preBlock->getParent();
    pr.getBlocks().insertAfter(preBlock->getIterator(),       dispatchHeader);
    pr.getBlocks().insertAfter(dispatchHeader->getIterator(), chunkBody);
    pr.getBlocks().insertAfter(chunkBody->getIterator(),      innerHeader);
    pr.getBlocks().insertAfter(innerHeader->getIterator(),    innerBody);
    pr.getBlocks().insertAfter(innerBody->getIterator(),      innerLatch);

    builder.setInsertionPointToEnd(preBlock);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, dispatchHeader);

    // dispatchHeader: hasWork = dispatch_next(...); loop while != 0.
    builder.setInsertionPointToEnd(dispatchHeader);
    SmallVector<Type> nextArgTys = {ptrT, iterTy, ptrT, ptrT, ptrT, ptrT};
    auto nextDecl = getOrInsertDeclWithReturn(module, dispatchNextFn,
      nextArgTys, iterTy, builder);
    Value hasWork = func::CallOp::create(builder, loc, nextDecl,
      ValueRange{identAddr, gtidVal, plast, plb, pub, pstride}).getResult(0);
    Value cond = LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne,
      hasWork, zero32);
    LLVM::CondBrOp::create(builder, loc, cond,
      chunkBody, mlir::ValueRange{}, afterAll, mlir::ValueRange{});

    // chunkBody: load this chunk's [lb, ub] (inclusive), init the inner IV.
    builder.setInsertionPointToEnd(chunkBody);
    Value cStart = LLVM::LoadOp::create(builder, loc, iterTy, plb);
    Value cEnd   = LLVM::LoadOp::create(builder, loc, iterTy, pub);
    Value piInner = LLVM::AllocaOp::create(builder, loc, ptrT, iterTy, one64);
    LLVM::StoreOp::create(builder, loc, cStart, piInner);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, innerHeader);

    // innerHeader: i <= ub ? innerBody : dispatchHeader  (inclusive, sle)
    builder.setInsertionPointToEnd(innerHeader);
    Value curI = LLVM::LoadOp::create(builder, loc, iterTy, piInner);
    Value innerCond = LLVM::ICmpOp::create(builder, loc,
      LLVM::ICmpPredicate::sle, curI, cEnd);
    LLVM::CondBrOp::create(builder, loc, innerCond,
      innerBody, mlir::ValueRange{}, dispatchHeader, mlir::ValueRange{});

    // innerBody: run the loop body.
    builder.setInsertionPointToEnd(innerBody);
    moveLoopBody(innerBody, innerLatch, curI);

    // innerLatch: i += step
    builder.setInsertionPointToEnd(innerLatch);
    Value nextI = LLVM::AddOp::create(builder, loc, curI, step);
    LLVM::StoreOp::create(builder, loc, nextI, piInner);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, innerHeader);

    // afterAll: post calls (barrier unless nowait).
    builder.setInsertionPointToStart(afterAll);
    emitPlanCalls(plan.post, builder);

    wsOp.erase();
    return;
  }

  // Choose loop bounds and comparison predicate based on how pre populated them.
  // Inline DIVMOD produces exclusive ubCore → use slt.
  // Runtime init (e.g. __kmpc_for_static_init_4) writes inclusive ub into pub
  // (we initialised pub with ub - step) → use sle.
  Value loopStart, loopEnd;
  LLVM::ICmpPredicate cmpPred;
  if (haveInlineBounds) {
    loopStart = lbCore;
    loopEnd   = ubCore;
    cmpPred   = LLVM::ICmpPredicate::slt;
  } else {
    loopStart = LLVM::LoadOp::create(builder, loc, iterTy, plb);
    loopEnd   = LLVM::LoadOp::create(builder, loc, iterTy, pub);
    cmpPred   = LLVM::ICmpPredicate::sle;
  }

  // -------------------------------------------------------------------------
  // Build the sequential loop.
  // -------------------------------------------------------------------------
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
  LLVM::StoreOp::create(builder, loc, loopStart, pi);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);

  builder.setInsertionPointToEnd(loopHeader);
  Value curI = LLVM::LoadOp::create(builder, loc, iterTy, pi);
  Value cond = LLVM::ICmpOp::create(builder, loc, cmpPred, curI, loopEnd);
  LLVM::CondBrOp::create(builder, loc, cond,
    loopBody, mlir::ValueRange{}, afterBlock, mlir::ValueRange{});

  // Move the loop nest body into loopBody and branch to loopLatch.
  moveLoopBody(loopBody, loopLatch, curI);

  builder.setInsertionPointToEnd(loopLatch);
  Value nextI = LLVM::AddOp::create(builder, loc, curI, step);
  LLVM::StoreOp::create(builder, loc, nextI, pi);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);

  // Emit post-loop calls (e.g. fini, optional barrier).
  builder.setInsertionPointToStart(afterBlock);
  emitPlanCalls(plan.post, builder);

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

    // Pre-compute the barrier plan once; the same plan is reused for every
    // omp.barrier inside outlined parallel functions.
    llvm::StringMap<dsl::Value> barrierCtx;
    barrierCtx["ident"]      = dsl::makeStr("%ident");
    barrierCtx["global_tid"] = dsl::makeStr("%tid");
    auto barrierPlan = evaluator.buildPlan(runtimeName, "barrier", barrierCtx);
    if (!barrierPlan) {
      module.emitError("omp-outline: barrier DSL evaluation failed: ")
        << llvm::toString(barrierPlan.takeError());
      return signalPassFailure();
    }

    // Step 1: outline parallel constructs.
    SmallVector<ConstructOp> constructs;
    module.walk([&](ConstructOp op) {
      if (!op.getBody().empty()) constructs.push_back(op);
    });
    int counter = 0;
    for (auto op : constructs)
      outlineConstruct(op, module, counter, *barrierPlan);

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
      ctx["schedule"]   = dsl::makeStr("static");
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
