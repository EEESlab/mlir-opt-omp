// Two passes in one: outlines omp_lower.construct bodies into func.func ops,
// and lowers omp.wsloop from its DSL plan.

#include "OmpLowering/Transforms/OmpOutliningPass.h"
#include "OmpLowering/IR/OmpLoweringOps.h"
#include "OmpLowering/DSL/DSLEvaluator.h"
#include "OmpLowering/DSL/DSLParser.h"
#include "PlanEmit.h"

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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"

#include <optional>

using namespace mlir;
using namespace mlir::omp_lower;

namespace {

static bool isPrivateCapture(Value val, Region &region) {
  auto allocaOp = val.getDefiningOp<LLVM::AllocaOp>();
  if (!allocaOp) return false;
  Type elemTy = allocaOp.getElemType();
  if (!LLVM::isCompatibleType(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMPointerType>(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMArrayType>(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMStructType>(elemTy)) return false;
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

// A scalar alloca read-only inside the region, so its value can be packed into
// the capture struct directly and save a dereference on every use.
static bool isScalarAllocaCapture(Value val, Region &region) {
  auto allocaOp = val.getDefiningOp<LLVM::AllocaOp>();
  if (!allocaOp) return false;
  Type elemTy = allocaOp.getElemType();
  if (!LLVM::isCompatibleType(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMPointerType>(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMArrayType>(elemTy)) return false;
  if (llvm::isa<LLVM::LLVMStructType>(elemTy)) return false;
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

// Partition captures into the three special-cased kinds.  Anything in none of
// them is captured as-is.  Order matters: each test excludes earlier claims.
static void classifyCaptures(ArrayRef<Value> captures, Region &region,
                             llvm::SetVector<Value> &privateCaptures,
                             llvm::SetVector<Value> &scalarAllocaCaptures,
                             llvm::SetVector<Value> &ptrAllocaCaptures) {
  for (auto cap : captures)
    if (isPrivateCapture(cap, region))
      privateCaptures.insert(cap);
  for (auto cap : captures)
    if (!privateCaptures.contains(cap) && isScalarAllocaCapture(cap, region))
      scalarAllocaCaptures.insert(cap);
  for (auto cap : captures)
    if (!privateCaptures.contains(cap) && !scalarAllocaCaptures.contains(cap) &&
        isPtrAllocaCapture(cap, region))
      ptrAllocaCaptures.insert(cap);
}

// A firstprivate value must be snapshotted into the capture struct at creation,
// not read through a pointer at entry, or a deferred task running after the
// source was mutated sees the wrong value.  classifyCaptures misses these.
static void forceFirstprivateByValue(
    Region &body, ArrayRef<Value> captures,
    const llvm::SetVector<Value> &privateCaptures,
    llvm::SetVector<Value> &scalarAllocaCaptures,
    const llvm::SetVector<Value> &ptrAllocaCaptures) {
  size_t numPriv = body.empty() ? 0 : body.front().getNumArguments();
  for (size_t i = 0; i < numPriv && i < captures.size(); i++) {
    Value src = captures[i];
    if (privateCaptures.contains(src) || scalarAllocaCaptures.contains(src) ||
        ptrAllocaCaptures.contains(src))
      continue;
    if (auto a = src.getDefiningOp<LLVM::AllocaOp>()) {
      Type et = a.getElemType();
      if (LLVM::isCompatibleType(et) &&
          !llvm::isa<LLVM::LLVMPointerType>(et) &&
          !llvm::isa<LLVM::LLVMArrayType>(et) &&
          !llvm::isa<LLVM::LLVMStructType>(et))
        scalarAllocaCaptures.insert(src);
    }
  }
}

static void replaceUsesInRegion(Region &region, Value oldVal, Value newVal) {
  for (auto &use : llvm::make_early_inc_range(oldVal.getUses()))
    if (region.isAncestor(use.getOwner()->getParentRegion()))
      use.set(newVal);
}

// Clause operands ride on the ConstructOp as one variadic list; the parallel
// clause_names attribute says which clause each entry is.
static Value getClauseOperand(ConstructOp op, llvm::StringRef name) {
  auto names = op.getClauseNames();
  if (!names) return Value();
  for (auto [i, n] : llvm::enumerate(*names))
    if (llvm::cast<StringAttr>(n).getValue() == name)
      return op.getClauseOperands()[i];
  return Value();
}

static std::string getPropStr(ConstructOp op, llvm::StringRef key) {
  auto dict = op.getPropDict();
  if (!dict) return "";
  if (auto sa = llvm::dyn_cast_or_null<StringAttr>(dict.get(key)))
    return sa.getValue().str();
  return "";
}

// capture_strategy is the single ABI discriminator; it entails the signature.
//   by_pointer -> microtask     void(gtid, btid, cap0, ...)
//   packed     -> closure       void(ptr data)
//   shareds    -> task routine  i32(gtid, ptr task), captures via task->shareds
enum class CaptureAbi { ByPointer, Packed, Shareds };

static std::optional<CaptureAbi> parseCaptureAbi(llvm::StringRef s) {
  if (s == "by_pointer") return CaptureAbi::ByPointer;
  if (s == "packed")     return CaptureAbi::Packed;
  if (s == "shareds")    return CaptureAbi::Shareds;
  return std::nullopt;
}

static LLVM::LLVMStructType getPropStructType(ConstructOp op, llvm::StringRef key,
    MLIRContext *ctx, LLVM::LLVMStructType fallback) {
  return parseStructProp(ctx, getPropStr(op, key), fallback);
}

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

// --- Capture-layout helpers (closure and task_entry) ---
// Both use the same packed struct; only where it lives differs.

static LLVM::LLVMStructType buildCaptureStruct(
    MLIRContext *ctx, ArrayRef<Value> captures,
    const llvm::SetVector<Value> &scalarAllocaCaptures,
    SmallVectorImpl<Type> &fieldTypes) {
  for (auto cap : captures) {
    if (scalarAllocaCaptures.contains(cap))
      fieldTypes.push_back(cap.getDefiningOp<LLVM::AllocaOp>().getElemType());
    else
      fieldTypes.push_back(cap.getType());
  }
  return LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);
}

static void unpackCapturesFromBase(
    OpBuilder &b, Location loc, Value one64, Value structBase,
    LLVM::LLVMStructType structTy, ArrayRef<Value> captures,
    const llvm::SetVector<Value> &privateCaptures,
    const llvm::SetVector<Value> &scalarAllocaCaptures,
    const llvm::SetVector<Value> &ptrAllocaCaptures, Region &fnBody,
    SmallVectorImpl<Value> *loadedCaptures = nullptr) {
  auto ptr = ptrTy(b.getContext());
  for (size_t i = 0; i < captures.size(); i++) {
    Value result;
    if (privateCaptures.contains(captures[i])) {
      // Private capture: a fresh per-thread alloca, not read from the struct.
      auto srcAlloca = captures[i].getDefiningOp<LLVM::AllocaOp>();
      result = LLVM::AllocaOp::create(b, loc, ptr, srcAlloca.getElemType(),
        one64);
      replaceUsesInRegion(fnBody, captures[i], result);
    } else if (scalarAllocaCaptures.contains(captures[i])) {
      // Load the scalar, then stash it in a fresh alloca so existing
      // load/store-of-alloca patterns keep working (SROA promotes it).
      Type elemTy = captures[i].getDefiningOp<LLVM::AllocaOp>().getElemType();
      Value gep = LLVM::GEPOp::create(b, loc, ptr, structTy, structBase,
        ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
      Value scalar = LLVM::LoadOp::create(b, loc, elemTy, gep);
      result = LLVM::AllocaOp::create(b, loc, ptr, elemTy, one64);
      LLVM::StoreOp::create(b, loc, scalar, result);
      replaceUsesInRegion(fnBody, captures[i], result);
    } else if (ptrAllocaCaptures.contains(captures[i])) {
      Value gep = LLVM::GEPOp::create(b, loc, ptr, structTy, structBase,
        ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
      Value pv = LLVM::LoadOp::create(b, loc, ptr, gep);
      result = LLVM::AllocaOp::create(b, loc, ptr, ptr, one64);
      LLVM::StoreOp::create(b, loc, pv, result);
      replaceUsesInRegion(fnBody, captures[i], result);
    } else {
      Value gep = LLVM::GEPOp::create(b, loc, ptr, structTy, structBase,
        ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
      result = LLVM::LoadOp::create(b, loc, captures[i].getType(), gep);
      replaceUsesInRegion(fnBody, captures[i], result);
    }
    if (loadedCaptures) loadedCaptures->push_back(result);
  }
}

// Compute the value each capture field receives; mirrors
// unpackCapturesFromBase.  Split from the store because the iomp task writes
// into a block the *plan* pass allocates, so these are handed over as bindings.
static void resolveCaptureValues(
    OpBuilder &b, Location loc, ArrayRef<Value> captures,
    ArrayRef<Type> fieldTypes,
    const llvm::SetVector<Value> &privateCaptures,
    const llvm::SetVector<Value> &scalarAllocaCaptures,
    const llvm::SetVector<Value> &ptrAllocaCaptures,
    SmallVectorImpl<Value> &out) {
  auto ptr = ptrTy(b.getContext());
  for (size_t i = 0; i < captures.size(); i++) {
    Value capVal;
    if (privateCaptures.contains(captures[i]))
      capVal = LLVM::UndefOp::create(b, loc, fieldTypes[i]);
    else if (scalarAllocaCaptures.contains(captures[i])) {
      Type elemTy = captures[i].getDefiningOp<LLVM::AllocaOp>().getElemType();
      capVal = LLVM::LoadOp::create(b, loc, elemTy, captures[i]);
    } else if (ptrAllocaCaptures.contains(captures[i]))
      capVal = LLVM::LoadOp::create(b, loc, ptr, captures[i]);
    else
      capVal = captures[i];
    out.push_back(capVal);
  }
}

static void storeCapturesToBase(
    OpBuilder &b, Location loc, Value structBase,
    LLVM::LLVMStructType structTy, ArrayRef<Value> captures,
    ArrayRef<Type> fieldTypes,
    const llvm::SetVector<Value> &privateCaptures,
    const llvm::SetVector<Value> &scalarAllocaCaptures,
    const llvm::SetVector<Value> &ptrAllocaCaptures) {
  SmallVector<Value> capVals;
  resolveCaptureValues(b, loc, captures, fieldTypes, privateCaptures,
                       scalarAllocaCaptures, ptrAllocaCaptures, capVals);
  auto ptr = ptrTy(b.getContext());
  for (size_t i = 0; i < capVals.size(); i++) {
    Value gep = LLVM::GEPOp::create(b, loc, ptr, structTy, structBase,
      ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
    LLVM::StoreOp::create(b, loc, capVals[i], gep);
  }
}

static Block &takeOutlinedBody(Region &body, func::FuncOp outlinedFn,
                               SmallVectorImpl<BlockArgument> &preexistingArgs) {
  // Strip private_vars before takeBody so replaceUsesInRegion cannot corrupt
  // those operand references.
  body.walk([&](Operation *walkOp) {
    if (auto wsOp = llvm::dyn_cast<omp::WsloopOp>(walkOp))
      if (!wsOp.getPrivateVars().empty())
        wsOp.getPrivateVarsMutable().clear();
    if (auto parOp = llvm::dyn_cast<omp::ParallelOp>(walkOp))
      if (!parOp.getPrivateVars().empty())
        parOp.getPrivateVarsMutable().clear();
  });

  outlinedFn.getBody().takeBody(body);
  Block &entry = outlinedFn.getBody().front();
  preexistingArgs.assign(entry.getArguments().begin(),
                         entry.getArguments().end());
  return entry;
}

static void eraseDeadCasts(func::FuncOp outlinedFn) {
  SmallVector<Operation *> casts;
  outlinedFn.getBody().walk([&](UnrealizedConversionCastOp c) {
    if (c->use_empty()) casts.push_back(c);
  });
  for (auto *c : casts) c->erase();
}

static void replaceTerminatorsWithReturn(
    func::FuncOp outlinedFn,
    llvm::function_ref<void(OpBuilder &, Location)> emitReturn) {
  SmallVector<Operation *> terms;
  for (auto &blk : outlinedFn.getBody())
    for (auto &innerOp : blk)
      if (innerOp.getName().getStringRef() == "omp.terminator")
        terms.push_back(&innerOp);
  for (auto *t : terms) {
    OpBuilder tb(t);
    emitReturn(tb, t->getLoc());
    t->erase();
  }
}

// --- 1a. IOMP TASK OUTLINING (capture_strategy = shareds) ---
// Outlines the construct into an entry i32(i32 gtid, ptr task) that loads
// task->shareds and unpacks the captures, leaving the construct standing with
// the entry pointer, ABI sizes and capture values bound to it.
// v1: pure private diagnosed not wired; no final; task_flags = 1 (tied).
static void bindGtidOnLeaves(func::FuncOp outlinedFn, MLIRContext *ctx,
                             llvm::function_ref<Value(OpBuilder &, Location)>
                                 makeGtid);
static bool planNamesToken(ConstructOp op, llvm::StringRef token);
static void warnIgnoredClauses(ConstructOp op);

static void outlineTaskEntry(ConstructOp op, ModuleOp module, int &counter) {
  Region &body = op.getBody();
  if (body.empty()) return;

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  auto ptr  = ptrTy(ctx);
  auto i32t = i32Ty(ctx);
  auto i64t = IntegerType::get(ctx, 64);

  SmallVector<Value> captures = collectCaptures(body);
  llvm::SetVector<Value> privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures;
  classifyCaptures(captures, body, privateCaptures, scalarAllocaCaptures,
                   ptrAllocaCaptures);

  forceFirstprivateByValue(body, captures, privateCaptures,
                           scalarAllocaCaptures, ptrAllocaCaptures);

  SmallVector<Type> fieldTypes;
  auto sharedsTy =
      buildCaptureStruct(ctx, captures, scalarAllocaCaptures, fieldTypes);

  // kmp_task_t header.  Only field 0 (shareds) is read here, but the full size
  // goes to the alloc call so the runtime's header writes stay in bounds.
  auto kmpTaskTy = getPropStructType(op, "kmp_task_t", ctx,
      LLVM::LLVMStructType::getLiteral(ctx, {ptr, ptr, i32t, ptr, ptr}));

  std::string fnName =
      "outlined_" + op.getConstructName().str() + "_" + std::to_string(counter++);

  OpBuilder builder(ctx);
  if (auto parentFn = op->getParentOfType<func::FuncOp>())
    builder.setInsertionPoint(parentFn);
  else
    builder.setInsertionPointToStart(module.getBody());

  auto outlinedFn = func::FuncOp::create(loc, fnName,
    FunctionType::get(ctx, {i32t, ptr}, {i32t}));
  outlinedFn.setVisibility(SymbolTable::Visibility::Nested);
  builder.insert(outlinedFn);

  // staleArgs receives the privatizer block args; the leftovers are dropped
  // after unpacking so the entry keeps its i32(i32 gtid, ptr task) ABI.
  SmallVector<BlockArgument> staleArgs;
  Block &entry = takeOutlinedBody(body, outlinedFn, staleArgs);

  entry.insertArgument(0u, i32t, loc);                         // gtid
  BlockArgument taskArg = entry.insertArgument(1u, ptr, loc);  // task

  if (!captures.empty()) {
    OpBuilder prologue(&entry, entry.begin());
    Value shGep = LLVM::GEPOp::create(prologue, loc, ptr, kmpTaskTy, taskArg,
      ArrayRef<LLVM::GEPArg>{0, 0});
    Value shareds = LLVM::LoadOp::create(prologue, loc, ptr, shGep);
    Value one64 = LLVM::ConstantOp::create(prologue, loc, i64t,
      IntegerAttr::get(i64t, 1));
    SmallVector<Value> loadedCaptures;
    unpackCapturesFromBase(prologue, loc, one64, shareds, sharedsTy, captures,
      privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures,
      outlinedFn.getBody(), &loadedCaptures);

    // firstprivate copy-in.  The source ptrs were injected at the start of the
    // region, so they lead the captures: loadedCaptures[i] pairs staleArgs[i].
    for (size_t i = 0; i < staleArgs.size() && i < loadedCaptures.size(); i++) {
      BlockArgument dstArg = staleArgs[i];
      Value srcPtr = loadedCaptures[i];
      if (!llvm::isa<LLVM::LLVMPointerType>(srcPtr.getType())) continue;
      Type elemTy;
      outlinedFn.getBody().walk([&](LLVM::LoadOp loadOp) {
        if (!elemTy && loadOp.getAddr() == dstArg)
          elemTy = loadOp.getRes().getType();
      });
      if (!elemTy || !LLVM::isCompatibleType(elemTy)) continue;
      Value privAlloca =
        LLVM::AllocaOp::create(prologue, loc, ptr, elemTy, one64);
      Value srcVal = LLVM::LoadOp::create(prologue, loc, elemTy, srcPtr);
      LLVM::StoreOp::create(prologue, loc, srcVal, privAlloca);
      replaceUsesInRegion(outlinedFn.getBody(), dstArg, privAlloca);
    }
  }

  // Drop the args the copy-in made dead.  A survivor with live uses is an
  // unsupported clause shape: diagnose rather than emit a wrong-ABI entry.
  for (auto arg : llvm::reverse(staleArgs)) {
    if (arg.use_empty())
      entry.eraseArgument(arg.getArgNumber());
    else
      op.emitError("omp-outline: iomp task has an unsupported private/"
                   "firstprivate clause; entry ABI would break");
  }

  eraseDeadCasts(outlinedFn);

  outlinedFn.setFunctionType(
      FunctionType::get(ctx, entry.getArgumentTypes(), {i32t}));

  replaceTerminatorsWithReturn(outlinedFn, [&](OpBuilder &tb, Location l) {
    Value zero = LLVM::ConstantOp::create(tb, l, i32t,
      IntegerAttr::get(i32t, 0));
    func::ReturnOp::create(tb, l, ValueRange{zero});
  });

  // A barrier is illegal in a task region — report it before binding below.
  outlinedFn.walk([&](ConstructOp inner) {
    if (inner.getConstructName() == "barrier")
      inner->emitError(
          "omp-outline: 'omp.barrier' is not valid inside a task region");
  });

  // A nested taskwait is lowered by the plan pass like any other leaf.  Here
  // %gtid is arg 0 *by value*, not the microtask's ptr-to-i32.
  bindGtidOnLeaves(outlinedFn, ctx, [&](OpBuilder &, Location) -> Value {
    return outlinedFn.getBody().front().getArgument(0);
  });

  // ---- Call site ----
  // Attach what only this pass can produce; ident and the thread id are left to
  // the plan pass, which materialises both on first reference.
  builder.setInsertionPoint(op);

  DataLayout dl(module);
  uint64_t taskSize    = dl.getTypeSize(kmpTaskTy).getFixedValue();
  uint64_t sharedsSize = dl.getTypeSize(sharedsTy).getFixedValue();

  Value fnPtr = func::ConstantOp::create(builder, loc,
    outlinedFn.getFunctionType(), FlatSymbolRefAttr::get(ctx, fnName));
  Value fnPtrCast = UnrealizedConversionCastOp::create(builder, loc,
    TypeRange{ptr}, ValueRange{fnPtr}).getResult(0);

  // Resolved here because the capture classification is outlining knowledge;
  // the plan pass only GEPs and stores them, in this order.
  SmallVector<Value> capVals;
  resolveCaptureValues(builder, loc, captures, fieldTypes, privateCaptures,
                       scalarAllocaCaptures, ptrAllocaCaptures, capVals);

  SmallVector<Value> operands(op.getClauseOperands().begin(),
                              op.getClauseOperands().end());
  SmallVector<Attribute> names;
  if (auto existing = op.getClauseNames())
    names.assign(existing->begin(), existing->end());
  auto bind = [&](llvm::StringRef n, Value v) {
    operands.push_back(v);
    names.push_back(StringAttr::get(ctx, n));
  };
  bind("body", fnPtrCast);
  bind("task_size", LLVM::ConstantOp::create(builder, loc, i64t,
                      IntegerAttr::get(i64t, (int64_t)taskSize)));
  bind("shareds_size", LLVM::ConstantOp::create(builder, loc, i64t,
                         IntegerAttr::get(i64t, (int64_t)sharedsSize)));
  // A list: every operand under this name belongs to it, and list_names keeps
  // it a list even when there are none.
  for (Value v : capVals) bind("%captures", v);

  op->setOperands(operands);
  op.setClauseNamesAttr(ArrayAttr::get(ctx, names));
  op.setListNamesAttr(ArrayAttr::get(ctx,
      {StringAttr::get(ctx, "%captures")}));
  warnIgnoredClauses(op);
  return;   // deliberately not erased: the plan pass consumes it
}

static bool blockNamesToken(ArrayAttr block, llvm::StringRef token) {
  if (!block) return false;
  auto is = [&](Attribute a) {
    auto s = llvm::dyn_cast<StringAttr>(a);
    return s && s.getValue() == token;
  };
  for (Attribute a : block) {
    if (auto call = llvm::dyn_cast<PlanCallAttr>(a)) {
      for (Attribute arg : call.getArgs())
        if (is(arg)) return true;
      continue;
    }
    if (auto br = llvm::dyn_cast<PlanBranchAttr>(a)) {
      if (is(br.getCond())) return true;
      if (blockNamesToken(br.getIfTrue(), token) ||
          blockNamesToken(br.getIfFalse(), token))
        return true;
    }
  }
  return false;
}

static bool planNamesToken(ConstructOp op, llvm::StringRef token) {
  return blockNamesToken(op.getPre(), token) ||
         blockNamesToken(op.getInvoke(), token) ||
         blockNamesToken(op.getPost(), token);
}

// A clause no plan action names has no lowering in this runtime's rules, and
// emitting it anyway would change semantics silently.  Per construct: only the
// plan knows which.
static void warnIgnoredClauses(ConstructOp op) {
  // The tokens a plan may name the clause by.  proc_bind has two — the push
  // argument (iomp) and the flags word GOMP always takes — and either counts.
  struct Clause { llvm::StringRef token, altToken, spelling; };
  static const Clause kClauses[] = {
    {"if_clause",   "",                "if"},
    {"num_threads", "",                "num_threads"},
    {"proc_bind",   "proc_bind_flags", "proc_bind"},
  };
  for (auto [token, altToken, spelling] : kClauses) {
    // proc_bind is the one clause with no SSA value: it rides as an attribute.
    bool present = token == "proc_bind" ? op.getProcBind().has_value()
                                        : (bool)getClauseOperand(op, token);
    bool named = planNamesToken(op, token) ||
                 (!altToken.empty() && planNamesToken(op, altToken));
    if (present && !named)
      op.emitWarning("omp-outline: `")
          << spelling
          << "` clause is not supported by this runtime/construct lowering "
             "and was ignored";
  }
}

static void bindGtidOnLeaves(func::FuncOp outlinedFn, MLIRContext *ctx,
                             llvm::function_ref<Value(OpBuilder &, Location)>
                                 makeGtid) {
  // Only leaves whose plan names %gtid.  Binding the rest would leave a dead
  // undef at the top of every closure-runtime outlined function.
  SmallVector<ConstructOp> leaves;
  outlinedFn.walk([&](ConstructOp c) {
    if (c.getBody().empty() && planNamesToken(c, "%gtid"))
      leaves.push_back(c);
  });
  if (leaves.empty()) return;

  auto &entry = outlinedFn.getBody().front();
  OpBuilder bb(ctx);
  bb.setInsertionPointToStart(&entry);
  Value gtid = makeGtid(bb, outlinedFn.getLoc());

  for (auto c : leaves) {
    SmallVector<Value> operands(c.getClauseOperands().begin(),
                                c.getClauseOperands().end());
    SmallVector<Attribute> names;
    if (auto existing = c.getClauseNames())
      names.assign(existing->begin(), existing->end());
    operands.push_back(gtid);
    names.push_back(StringAttr::get(ctx, "%gtid"));
    c->setOperands(operands);
    c.setClauseNamesAttr(ArrayAttr::get(ctx, names));
  }
}


static void outlineConstruct(ConstructOp op, ModuleOp module, int &counter) {
  Region &body = op.getBody();
  if (body.empty()) return;

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();

  std::string captureStrat = getPropStr(op, "capture_strategy");
  std::optional<CaptureAbi> abiOpt = parseCaptureAbi(captureStrat);
  if (!abiOpt) {
    op.emitError("unknown capture_strategy '" + captureStrat +
                 "' (expected by_pointer, packed, or shareds)");
    return;
  }
  CaptureAbi abi = *abiOpt;

  if (abi == CaptureAbi::Shareds) {
    outlineTaskEntry(op, module, counter);
    return;
  }

  SmallVector<Value> captures = collectCaptures(body);

  // Classify before takeBody (which consumes the region).  Scalar- and
  // ptr-alloca captures are packed by value under the packed strategy, removing
  // a level of indirection on every use and unblocking LICM.
  llvm::SetVector<Value> privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures;
  classifyCaptures(captures, body, privateCaptures, scalarAllocaCaptures,
                   ptrAllocaCaptures);

  // Only the packed path consumes these buckets; by_pointer passes captures as
  // individual args.
  if (abi == CaptureAbi::Packed)
    forceFirstprivateByValue(body, captures, privateCaptures,
                             scalarAllocaCaptures, ptrAllocaCaptures);

  std::string fnName =
      "outlined_" + op.getConstructName().str() + "_" + std::to_string(counter++);

  SmallVector<Type> fnArgTypes;
  if (abi == CaptureAbi::ByPointer) {
    fnArgTypes.push_back(ptrTy(ctx)); // ptr gtid
    fnArgTypes.push_back(ptrTy(ctx)); // ptr btid
    for (auto cap : captures) fnArgTypes.push_back(cap.getType());
  } else {
    fnArgTypes.push_back(ptrTy(ctx));
  }

  OpBuilder builder(ctx);
  if (auto parentFn = op->getParentOfType<func::FuncOp>())
    builder.setInsertionPoint(parentFn);
  else
    builder.setInsertionPointToStart(module.getBody());

  auto outlinedFn = func::FuncOp::create(loc, fnName,
    FunctionType::get(ctx, fnArgTypes, {}));
  // Nested visibility: the function is referenced via func.constant, which
  // --remove-dead-values may not track after lowering.
  outlinedFn.setVisibility(SymbolTable::Visibility::Nested);
  builder.insert(outlinedFn);

  SmallVector<BlockArgument> privatizerArgs;
  Block &entry = takeOutlinedBody(body, outlinedFn, privatizerArgs);

  if (abi == CaptureAbi::Packed) {
    // --- PACKED / CLOSURE (libgomp): one ptr to a capture struct, arg 0 ---
    entry.insertArgument(0u, ptrTy(ctx), loc);
    BlockArgument dataPtr = entry.getArgument(0u);

    SmallVector<Type> fieldTypes;
    auto structTy =
        buildCaptureStruct(ctx, captures, scalarAllocaCaptures, fieldTypes);

    if (!captures.empty()) {
      OpBuilder prologue(&entry, entry.begin());
      Value one64 = LLVM::ConstantOp::create(prologue, loc,
        IntegerType::get(ctx, 64),
        IntegerAttr::get(IntegerType::get(ctx, 64), 1));
      SmallVector<Value> loadedCaptures;
      unpackCapturesFromBase(prologue, loc, one64, dataPtr, structTy, captures,
        privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures,
        outlinedFn.getBody(), &loadedCaptures);
      // firstprivate: the source ptrs were already loaded above into
      // loadedCaptures — reuse them rather than making new GEPs.
      size_t numPriv = privatizerArgs.size();
      // The injected marker casts sit at the START of the entry block, so
      // collectCaptures found them first: indices 0..numPriv-1.
      size_t privCapStart = 0;
      for (size_t i = 0; i < numPriv; i++) {
        BlockArgument dstArg = privatizerArgs[i];
        Value srcPtr = loadedCaptures[privCapStart + i]; // already loaded ptr
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
  } else if (abi == CaptureAbi::ByPointer) {
    // --- BY_POINTER (iomp microtask): one arg per capture ---
    for (int i = (int)captures.size() - 1; i >= 0; i--) {
      BlockArgument arg = entry.insertArgument(0u, captures[i].getType(), loc);
      replaceUsesInRegion(outlinedFn.getBody(), captures[i], arg);
    }
    entry.insertArgument(0u, ptrTy(ctx), loc); // btid
    entry.insertArgument(0u, ptrTy(ctx), loc); // gtid

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
        Type elemTy;
        outlinedFn.getBody().walk([&](LLVM::LoadOp loadOp) {
          if (!elemTy && loadOp.getAddr() == dstArg)
            elemTy = loadOp.getRes().getType();
        });
        if (!elemTy) {
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
        // The original privatizer arg loads used alignment=1.
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
  }

  // Drop the args the copy-in made dead; a live survivor is an unsupported shape.
  if (!privatizerArgs.empty()) {
    for (auto arg : llvm::reverse(privatizerArgs)) {
      if (arg.use_empty())
        entry.eraseArgument(arg.getArgNumber());
      else
        op.emitError("omp-outline: ")
            << op.getConstructName()
            << " construct has an unsupported private/firstprivate clause; "
               "outlined ABI would break";
    }
  }

  eraseDeadCasts(outlinedFn);

  {
    unsigned capBase = abi == CaptureAbi::ByPointer ? 2 : 1; // microtask: [gtid,btid,...]; packed: [data,...]
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

  SmallVector<Type> finalArgTypes;
  for (auto arg : entry.getArguments())
    finalArgTypes.push_back(arg.getType());
  outlinedFn.setFunctionType(FunctionType::get(ctx, finalArgTypes, {}));

  // Mark capture pointers noalias.  Each arrives as a pointer to a distinct
  // caller-side slot, so a store through one never touches another's storage.
  // Without it the optimiser must assume they alias, blocking LICM and
  // vectorisation.  gtid/btid are runtime-owned (captures start at index 2).
  if (abi == CaptureAbi::ByPointer) {
    UnitAttr noalias = UnitAttr::get(ctx);
    for (unsigned i = 2, e = entry.getNumArguments(); i < e; i++)
      if (llvm::isa<LLVM::LLVMPointerType>(entry.getArgument(i).getType()))
        outlinedFn.setArgAttr(i, "llvm.noalias", noalias);
  }

  replaceTerminatorsWithReturn(outlinedFn, [](OpBuilder &tb, Location l) {
    func::ReturnOp::create(tb, l);
  });

  // Bind the thread id the leaf constructs cannot derive.  This is the microtask
  // entry, so %gtid is a load of arg 0, materialised once so it dominates them.
  bindGtidOnLeaves(outlinedFn, ctx, [&](OpBuilder &bb, Location bloc) -> Value {
    auto &fnEntry = outlinedFn.getBody().front();
    if (fnEntry.getNumArguments() >= 2)
      return LLVM::LoadOp::create(bb, bloc, i32Ty(ctx), fnEntry.getArgument(0));
    return LLVM::UndefOp::create(bb, bloc, i32Ty(ctx));
  });

  // ---- Hand the construct to PlanLoweringPass ----
  // No runtime call is emitted here; only what this pass alone can produce.
  builder.setInsertionPoint(op);

  SmallVector<Value> operands(op.getClauseOperands().begin(),
                              op.getClauseOperands().end());
  SmallVector<Attribute> names;
  if (auto existing = op.getClauseNames())
    names.assign(existing->begin(), existing->end());
  auto bind = [&](llvm::StringRef n, Value v) {
    operands.push_back(v);
    names.push_back(StringAttr::get(ctx, n));
  };

  // Under every spelling a plan may use: the context seeds body for task and
  // outlined_parallel for parallel, and a plan can also name it as a callee.
  Value fnPtr = func::ConstantOp::create(builder, loc,
    outlinedFn.getFunctionType(), FlatSymbolRefAttr::get(ctx, fnName));
  Value fnPtrCast = UnrealizedConversionCastOp::create(
    builder, loc, TypeRange{ptrTy(ctx)}, ValueRange{fnPtr}).getResult(0);
  bind("body", fnPtrCast);
  bind("outlined_parallel", fnPtrCast);
  bind("outlined_task", fnPtrCast);

  // proc_bind is compile-time, so it arrives as an attribute and its constant is
  // made here.  Two tokens carry it: proc_bind (the clause) and proc_bind_flags
  // (the word GOMP always takes, 0 if none).  Each bound only if the plan names it.
  uint32_t affinity = 0;  // kmp_proc_bind_false / omp_proc_bind_false
  if (auto procBind = op.getProcBind()) {
    if (auto value = procBindEnumValue(*procBind))
      affinity = *value;
    else
      op.emitError("omp-outline: unknown proc_bind kind '")
          << *procBind << "'; no affinity constant to emit";
  }
  auto affinityConst = [&] {
    return LLVM::ConstantOp::create(builder, loc, i32Ty(ctx),
             IntegerAttr::get(i32Ty(ctx), (int64_t)affinity));
  };
  if (op.getProcBind() && planNamesToken(op, "proc_bind"))
    bind("proc_bind", affinityConst());
  if (planNamesToken(op, "proc_bind_flags"))
    bind("proc_bind_flags", affinityConst());

  auto i64t = IntegerType::get(ctx, 64);
  auto one64 = [&] {
    return LLVM::ConstantOp::create(builder, loc, i64t,
                                    IntegerAttr::get(i64t, 1));
  };

  if (abi == CaptureAbi::Packed) {
    // --- PACKED / CLOSURE: one capture struct on the stack, passed by ptr ---
    // The struct type also gives env_size / env_align.
    SmallVector<Type> fieldTypes;
    auto structTy =
        buildCaptureStruct(ctx, captures, scalarAllocaCaptures, fieldTypes);

    Value structAlloca;
    if (!captures.empty()) {
      structAlloca = LLVM::AllocaOp::create(builder, loc,
        ptrTy(ctx), structTy, one64());
      storeCapturesToBase(builder, loc, structAlloca, structTy, captures,
        fieldTypes, privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures);
    } else {
      structAlloca = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
    }
    bind("env_ptr", structAlloca);

    // GOMP_task takes these as long arg_size, long arg_align (libgomp memcpys
    // arg_size bytes when cpyfn is NULL).  Alignment falls back to 16.
    DataLayout dataLayout(module);
    uint64_t envSize  = captures.empty()
      ? 0u : dataLayout.getTypeSize(structTy).getFixedValue();
    uint64_t envAlign = captures.empty()
      ? 1u : dataLayout.getTypeABIAlignment(structTy);
    if (envAlign == 0) envAlign = 16;
    bind("env_size",  LLVM::ConstantOp::create(builder, loc, i64t,
                        IntegerAttr::get(i64t, (int64_t)envSize)));
    bind("env_align", LLVM::ConstantOp::create(builder, loc, i64t,
                        IntegerAttr::get(i64t, (int64_t)envAlign)));
  } else {
    // --- BY_POINTER (iomp microtask): each capture is its own trailing arg ---
    // A private capture (a loop IV) gets a fresh alloca per thread.
    SmallVector<Value> capVals;
    for (auto cap : captures) {
      Value capVal = cap;
      if (privateCaptures.contains(cap)) {
        auto srcAlloca = cap.getDefiningOp<LLVM::AllocaOp>();
        capVal = LLVM::AllocaOp::create(builder, loc, ptrTy(ctx),
          srcAlloca.getElemType(), one64());
      }
      capVals.push_back(capVal);
    }

    // The serialized side of if calls the microtask directly, and that ABI takes
    // gtid and btid *by pointer* — two slots only this pass can make.
    if (getClauseOperand(op, "if_clause")) {
      std::string gtidFnName = getPropStr(op, "global_tid_function");
      Value gtid;
      if (gtidFnName.empty()) {
        // Declaring a nameless function would fail the verifier.
        op.emitWarning("'%gtid' referenced but the runtime defines no "
                       "global_tid_function; using undef");
        gtid = LLVM::UndefOp::create(builder, loc, i32Ty(ctx));
      } else {
        Value ident = getOrCreateIdent(module, builder, loc, ctx, kIdentKmpc);
        auto gtidDecl = getOrInsertDeclWithReturn(module, gtidFnName,
          {ptrTy(ctx)}, i32Ty(ctx), builder);
        gtid = func::CallOp::create(builder, loc, gtidDecl,
          ValueRange{ident}).getResult(0);
      }
      Value zero32 = LLVM::ConstantOp::create(builder, loc, i32Ty(ctx),
        IntegerAttr::get(i32Ty(ctx), 0));
      Value gtidAddr = LLVM::AllocaOp::create(builder, loc, ptrTy(ctx),
        i32Ty(ctx), one64());
      Value btidAddr = LLVM::AllocaOp::create(builder, loc, ptrTy(ctx),
        i32Ty(ctx), one64());
      LLVM::StoreOp::create(builder, loc, gtid, gtidAddr);
      LLVM::StoreOp::create(builder, loc, zero32, btidAddr);
      bind("gtid_addr", gtidAddr);
      bind("btid_addr", btidAddr);
      // The serialized calls need it and the store above already materialised
      // it; without this the plan pass would emit a second gtid call.
      bind("%gtid", gtid);
    }

    for (Value v : capVals) bind("%captures", v);
    op.setListNamesAttr(ArrayAttr::get(ctx,
        {StringAttr::get(ctx, "%captures")}));
  }

  op->setOperands(operands);
  op.setClauseNamesAttr(ArrayAttr::get(ctx, names));

  warnIgnoredClauses(op);
}

// One scratch slot per loop, in the entry block where the frame is fixed.  An
// alloca elsewhere is a *dynamic* allocation: inside a sequential loop around
// the wsloop the stack pointer walks down once per iteration and never comes
// back, overrunning a PULP cluster core's few kilobytes of stack.
static Value loopScratchSlot(OpBuilder &builder, Operation *anchor,
                             Location loc, Type ptrT, Type elemTy) {
  OpBuilder::InsertionGuard guard(builder);

  // Outlined bodies are func.func; anything else keeps the old placement.
  if (auto fn = anchor->getParentOfType<func::FuncOp>())
    if (!fn.getBody().empty())
      builder.setInsertionPointToStart(&fn.getBody().front());

  Value one = LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                       builder.getI64IntegerAttr(1));
  return LLVM::AllocaOp::create(builder, loc, ptrT, elemTy, one);
}

// The same rule for the allocas we did not write: the front-end puts a local's
// slot in the scope that declares it.  The PULP path runs opt with no passes, so
// nothing promotes them.  Hoisting a constant-size alloca is what LLVM's own
// inliner does; a variable-size one really is dynamic and stays.
static void hoistStaticAllocas(func::FuncOp fn) {
  if (fn.getBody().empty()) return;
  Block &entry = fn.getBody().front();

  SmallVector<LLVM::AllocaOp> movable;
  fn.getBody().walk([&](LLVM::AllocaOp alloca) {
    if (alloca->getBlock() == &entry) return;
    if (alloca.getArraySize().getDefiningOp<LLVM::ConstantOp>())
      movable.push_back(alloca);
  });

  for (auto alloca : movable) {
    // The size constant moves first, or the alloca would read a value defined
    // after it.  A constant takes no operands, so this is always safe.
    auto size = alloca.getArraySize().getDefiningOp<LLVM::ConstantOp>();
    if (size->getBlock() != &entry)
      size->moveBefore(&entry, entry.begin());
    alloca->moveAfter(size);
  }
}

// Returns failure once a diagnostic has been emitted: a wsloop left standing in
// the output is not something the rest of the pipeline can make sense of.
static LogicalResult lowerWsloop(omp::WsloopOp wsOp,
                                 ModuleOp module,
                                 const dsl::LoweringPlan &plan) {
  omp::LoopNestOp loopNest;
  wsOp.walk([&](omp::LoopNestOp op) { loopNest = op; });
  if (!loopNest) return success();

  MLIRContext *ctx = wsOp.getContext();
  Location loc = wsOp.getLoc();
  OpBuilder builder(wsOp);

  auto lbs   = loopNest.getLoopLowerBounds();
  auto ubs   = loopNest.getLoopUpperBounds();
  auto steps = loopNest.getLoopSteps();
  if (lbs.empty() || ubs.empty() || steps.empty()) return success();

  // Only the outermost dimension is distributed.  With a collapsed nest the
  // inner IVs keep their uses after the body moves out, so erasing would abort.
  if (lbs.size() > 1) {
    wsOp.emitError("omp-outline: a collapsed loop nest (")
        << lbs.size() << " dimensions) is not supported";
    return failure();
  }

  Value lb = lbs[0], ub = ubs[0], step = steps[0];
  Type iterTy = lb.getType();

  // Normalise to an exclusive upper bound so all downstream trip-count
  // arithmetic is uniform: exclusive_ub = inclusive_ub + step.
  if (loopNest.getLoopInclusive())
    ub = LLVM::AddOp::create(builder, loc, ub, step);

  // The thread id, only when the plan asks: on the closure runtimes it costs an
  // omp_get_thread_num call, which LLVM cannot delete for being unused.
  auto planNames = [&plan](llvm::StringRef token) {
    auto inBlock = [&](const std::vector<dsl::PlanAction> &actions) {
      for (auto &a : actions)
        if (auto *ca = std::get_if<dsl::PlanCall>(&a))
          for (auto &av : ca->args)
            if (auto *sv = std::get_if<dsl::StrVal>(&av))
              if (sv->value == token) return true;
      return false;
    };
    return inBlock(plan.pre) || inBlock(plan.invoke) || inBlock(plan.post) ||
           inBlock(plan.firstChunk) || inBlock(plan.nextChunk);
  };
  bool needsGtid = planNames("%gtid");

  Value gtidVal;
  if (needsGtid) gtidVal = LLVM::UndefOp::create(builder, loc, iterTy);
  if (auto parentFn = wsOp->getParentOfType<func::FuncOp>(); parentFn && needsGtid) {
    auto &entry = parentFn.getBody().front();
    unsigned numArgs = entry.getNumArguments();
    // Microtask: >= 2 args (gtid ptr, btid ptr).  Closure: exactly 1 (data ptr).
    bool isMicrotaskFn = numArgs >= 2 &&
      llvm::isa<LLVM::LLVMPointerType>(entry.getArgument(0).getType()) &&
      llvm::isa<LLVM::LLVMPointerType>(entry.getArgument(1).getType());
    if (isMicrotaskFn) {
      Value gtidPtr = entry.getArgument(0);
      gtidVal = LLVM::LoadOp::create(builder, loc, iterTy, gtidPtr);
    } else {
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
  Value zero32 = LLVM::ConstantOp::create(builder, loc, iterTy,
    IntegerAttr::get(iterTy, 0));
  Value one32 = LLVM::ConstantOp::create(builder, loc, iterTy,
    IntegerAttr::get(iterTy, 1));

  auto getStrProp = [&](llvm::StringRef key) -> std::string {
    auto it = plan.properties.find(key.str());
    if (it == plan.properties.end()) return "";
    if (auto *sv = std::get_if<dsl::StrVal>(&it->second)) return sv->value;
    return "";
  };

  // A property present but unreadable is a typo in the rules, and each fails a
  // different way if it quietly falls back: wrong slot width, wrong truthiness
  // test, or an inner loop running one past the end of every chunk.
  bool badProp = false;
  auto typeProp = [&](llvm::StringRef key, Type fallback) -> Type {
    std::string spelling = getStrProp(key);
    if (spelling.empty()) return fallback;
    if (Type t = parseAbiTypeProp(ctx, spelling, Type())) return t;
    wsOp.emitError("omp-outline: `") << key << " = " << spelling
        << "` is not a type this lowering knows (i8, i32, i64, ptr)";
    badProp = true;
    return fallback;
  };

  // The index type of this runtime's loop ABI, not always the loop's own:
  // libgomp's GOMP_loop_* is long-based whatever the IV is, iomp's __kmpc_*_4
  // matches an i32.  Defaulting to the IV's type avoids conversions elsewhere.
  Type chunkIdxTy = typeProp("chunk_index", iterTy);

  // A next_chunk block means the runtime hands out iterations a chunk at a time.
  // Which schedules are chunked is a statement rules.dsl makes, not this pass.
  bool isChunked = !plan.nextChunk.empty();
  // What such a call returns (0 = no work left): an int for iomp, a _Bool for GOMP.
  Type chunkResTy = typeProp("chunk_result", i32Ty(ctx));

  // Whether the upper bound the runtime writes is the last valid iteration or
  // the one past it.  Read here so a misspelling is caught with the other two.
  std::string chunkBound = getStrProp("chunk_bound");
  if (!chunkBound.empty() && chunkBound != "inclusive" &&
      chunkBound != "exclusive") {
    wsOp.emitError("omp-outline: `chunk_bound = ")
        << chunkBound << "` is neither `inclusive` nor `exclusive`";
    badProp = true;
  }

  // The chunk size comes from the clause, so its type is the input's to choose.
  // A non-integer has no conversion into the index type and would abort below.
  Value chunkVal = wsOp.getScheduleChunk();
  if (chunkVal && !llvm::isa<IntegerType>(chunkVal.getType())) {
    wsOp.emitError("omp-outline: the schedule chunk must be an integer, got ")
        << chunkVal.getType();
    badProp = true;
  }

  // first_chunk without next_chunk leaves isChunked false, so the block is never
  // emitted: the work-share never registers.  Wrong code, quietly — refuse it.
  if (!plan.firstChunk.empty() && plan.nextChunk.empty()) {
    wsOp.emitError("omp-outline: this wsloop declares `first_chunk` but no "
                   "`next_chunk`; there is no call to ask for another chunk");
    badProp = true;
  }

  if (badProp) return failure();

  // Loop indices are signed, so widening is a sign extension.  Both sides are
  // integers by the checks above.
  auto convertInt = [&](Value v, Type to) -> Value {
    if (v.getType() == to) return v;
    auto fromTy = llvm::dyn_cast<IntegerType>(v.getType());
    auto destTy = llvm::dyn_cast<IntegerType>(to);
    if (!fromTy || !destTy) return v;
    return destTy.getWidth() > fromTy.getWidth()
               ? LLVM::SExtOp::create(builder, loc, to, v).getResult()
               : LLVM::TruncOp::create(builder, loc, to, v).getResult();
  };
  auto toIdx   = [&](Value v) { return convertInt(v, chunkIdxTy); };
  auto fromIdx = [&](Value v) { return convertInt(v, iterTy); };

  // In the entry block, see loopScratchSlot.  Element type is the ABI's index
  // type: the runtime writes them.  pstride is pure output, so not initialised.
  Value plb     = loopScratchSlot(builder, wsOp, loc, ptrT, chunkIdxTy);
  Value pub     = loopScratchSlot(builder, wsOp, loc, ptrT, chunkIdxTy);
  Value pstride = loopScratchSlot(builder, wsOp, loc, ptrT, chunkIdxTy);
  Value plast   = loopScratchSlot(builder, wsOp, loc, ptrT, chunkIdxTy);

  // The last valid iteration, for the entry points taking the bound that way
  // (the __kmpc_* family; GOMP's does not).  NOT ub - step: the step need not
  // divide the range.  0 to 10 by 3 ends at 9, where ub - step is 7 — handing
  // that over loses the last iteration.  Take the trip count first, then walk
  // back one step.  Only when asked for: five ops rather than one.
  Value ubInclusive;
  if (!isChunked || planNames("%ub_incl")) {
    Value range   = LLVM::SubOp::create(builder, loc, ub, lb);
    Value rangeUp = LLVM::AddOp::create(builder, loc, range,
                      LLVM::SubOp::create(builder, loc, step, one32));
    Value trip    = LLVM::SDivOp::create(builder, loc, rangeUp, step);
    Value lastIdx = LLVM::SubOp::create(builder, loc, trip, one32);
    ubInclusive   = LLVM::AddOp::create(builder, loc, lb,
                      LLVM::MulOp::create(builder, loc, lastIdx, step));
  }

  // Chunked: the slots are pure output — bounds go in by value through the
  // acquisition call — so there is nothing to seed.
  if (!isChunked) {
    LLVM::StoreOp::create(builder, loc, toIdx(lb),          plb);
    LLVM::StoreOp::create(builder, loc, toIdx(ubInclusive), pub);
    LLVM::StoreOp::create(builder, loc, toIdx(zero32),      plast);
  }

  // The loop-specific site bindings; ident and %gtid come from the shared
  // vocabulary (see resolveSymbolToken).
  llvm::StringMap<Value> wsBindings;
  wsBindings["%lb"]     = plb;      // in/out lower-bound slot
  wsBindings["%ub"]     = pub;      // in/out upper-bound slot
  wsBindings["%step"]   = step;     // actual loop step
  wsBindings["%stride"] = pstride;  // output ptr for runtime stride
  wsBindings["%last"]   = plast;    // in/out last-iteration flag
  // The bounds by value, for the dispatch APIs taking them that way.  Which
  // upper bound is its own ABI: iomp the last valid iteration, libgomp the next.
  wsBindings["%lb_val"]  = lb;
  wsBindings["%ub_val"]  = ub;  // exclusive, as loop_nest was normalised
  if (ubInclusive) wsBindings["%ub_incl"] = ubInclusive;  // last valid iteration
  if (chunkVal) wsBindings["%chunk"] = chunkVal;  // integer, checked above

  // Everything naming a loop index crosses into the runtime's index type;
  // pointers and the thread id do not.  Identity for every runtime but libgomp.
  auto isIndexToken = [](llvm::StringRef t) {
    return t == "%lb_val" || t == "%ub_val" || t == "%ub_incl" ||
           t == "%step"   || t == "%chunk";
  };
  auto resolveCallArg = [&](const dsl::Value &v) -> Value {
    if (auto *sv = std::get_if<dsl::StrVal>(&v)) {
      Value r = resolveSymbolToken(
          sv->value, builder, loc, wsBindings,
          [&](uint32_t flags) {
            return getOrCreateIdent(module, builder, loc, ctx, flags);
          },
          [&] { return gtidVal; });
      return isIndexToken(sv->value) ? toIdx(r) : r;
    }
    if (auto *iv = std::get_if<dsl::IntVal>(&v))
      return LLVM::ConstantOp::create(builder, loc, chunkIdxTy,
        IntegerAttr::get(chunkIdxTy, iv->value));
    return LLVM::UndefOp::create(builder, loc, ptrT);
  };

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

  // plan.pre: emit the PlanCalls; emit thread_bounds instead materialises
  // per-thread [lbThread, ubThread) inline.  Which one ran drives the bounds
  // choice below.
  Value lbThread, ubThread;
  bool haveInlineBounds = false;
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
    if (!pe || pe->name != "thread_bounds") continue;

    // Block distribution, one contiguous chunk per thread:
    //   trip     = ceil((ub - lb) / step)
    //   chunk    = ceil(trip / num_threads)
    //   lbThread = lb + threadId * chunk * step
    //   ubThread = min(lbThread + chunk * step, ub)
    // One ceiling division and a clamping select, not a per-thread SDiv+SRem.
    std::string threadIdFn  = getStrProp("thread_id_function");
    std::string numThreadFn = getStrProp("num_threads_function");
    Value threadId   = emitNoArgI32Call(module, builder, loc, threadIdFn);
    Value numThreads = emitNoArgI32Call(module, builder, loc, numThreadFn);

    Value range    = LLVM::SubOp::create(builder, loc, ub, lb);
    Value rangeS   = LLVM::AddOp::create(builder, loc, range,
                       LLVM::SubOp::create(builder, loc, step, one32));
    Value trip     = LLVM::SDivOp::create(builder, loc, rangeS, step);

    Value tripPlusNC = LLVM::AddOp::create(builder, loc, trip,
                         LLVM::SubOp::create(builder, loc, numThreads, one32));
    Value chunk    = LLVM::SDivOp::create(builder, loc, tripPlusNC, numThreads);

    Value threadChunk  = LLVM::MulOp::create(builder, loc, threadId, chunk);
    Value threadOff    = LLVM::MulOp::create(builder, loc, threadChunk, step);
    lbThread           = LLVM::AddOp::create(builder, loc, lb, threadOff);

    Value chunkStep  = LLVM::MulOp::create(builder, loc, chunk, step);
    Value lbThreadEnd  = LLVM::AddOp::create(builder, loc, lbThread, chunkStep);
    Value clampCond  = LLVM::ICmpOp::create(builder, loc,
                         LLVM::ICmpPredicate::sgt, lbThreadEnd, ub);
    ubThread           = LLVM::SelectOp::create(builder, loc, clampCond, ub, lbThreadEnd);
    haveInlineBounds = true;
  }

  // Bounds and predicate follow how pre populated them: inline chunking gives an
  // exclusive ubThread (slt), a runtime init an inclusive one (sle).  A chunked
  // construct has neither yet, so the predicate is the runtime's convention.
  Value loopStart, loopEnd;
  LLVM::ICmpPredicate cmpPred = LLVM::ICmpPredicate::sle;
  if (haveInlineBounds) {
    loopStart = lbThread;
    loopEnd   = ubThread;
    cmpPred   = LLVM::ICmpPredicate::slt;
  } else if (!isChunked) {
    loopStart = LLVM::LoadOp::create(builder, loc, iterTy, plb);
    loopEnd   = LLVM::LoadOp::create(builder, loc, iterTy, pub);
    cmpPred   = LLVM::ICmpPredicate::sle;
  } else {
    cmpPred = chunkBound == "exclusive" ? LLVM::ICmpPredicate::slt
                                        : LLVM::ICmpPredicate::sle;
  }

  // Emit a chunk-acquisition block and give back the i1 there-was-work test of
  // its last call — a when/otherwise chain collapses to exactly one.
  auto emitChunkCall = [&](const std::vector<dsl::PlanAction> &actions) -> Value {
    Value more;
    for (auto &action : actions) {
      auto *ca = std::get_if<dsl::PlanCall>(&action);
      if (!ca) continue;
      SmallVector<Value> args;
      SmallVector<Type>  types;
      for (auto &av : ca->args) {
        Value v = resolveCallArg(av);
        args.push_back(v); types.push_back(v.getType());
      }
      auto decl =
          getOrInsertDeclWithReturn(module, ca->callee, types, chunkResTy, builder);
      more = func::CallOp::create(builder, loc, decl, args).getResult(0);
    }
    if (!more) return Value();
    Value zero = LLVM::ConstantOp::create(builder, loc, chunkResTy,
      IntegerAttr::get(chunkResTy, 0));
    return LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne, more, zero);
  };

  // A chunk block with no call leaves the loop with nothing to turn on.
  auto blockHasCall = [](const std::vector<dsl::PlanAction> &b) {
    for (auto &a : b)
      if (std::get_if<dsl::PlanCall>(&a)) return true;
    return false;
  };
  if (isChunked && (!blockHasCall(plan.nextChunk) ||
                    (!plan.firstChunk.empty() && !blockHasCall(plan.firstChunk)))) {
    wsOp.emitError("omp-outline: the chunk block of this wsloop makes no call; "
                   "there is no chunk to loop on");
    return failure();
  }

  // Build the loop.  A chunked construct wraps the sequential loop in an outer
  // one, rotated so the acquisition call sits in a guard and a latch:
  //   pre:        first_chunk? -> chunkBody : after
  //   chunkBody:  read the slots the call filled, then the inner loop
  //   chunkLatch: next_chunk?  -> chunkBody : after
  // Rotating it lets the opening call differ from the repeat one, which libgomp
  // needs and iomp does not.
  Block *preBlock   = builder.getInsertionBlock();
  Block *afterBlock = preBlock->splitBlock(builder.getInsertionPoint());
  Block *chunkBody  = isChunked ? new Block() : nullptr;
  Block *loopHeader = new Block();
  Block *loopBody   = new Block();
  Block *loopLatch  = new Block();
  Block *chunkLatch = isChunked ? new Block() : nullptr;

  auto &parentRegion = *preBlock->getParent();
  auto lastInserted = preBlock->getIterator();
  auto insertBlock = [&](Block *b) {
    parentRegion.getBlocks().insertAfter(lastInserted, b);
    lastInserted = b->getIterator();
  };
  if (chunkBody) insertBlock(chunkBody);
  insertBlock(loopHeader);
  insertBlock(loopBody);
  insertBlock(loopLatch);
  if (chunkLatch) insertBlock(chunkLatch);

  // The induction slot goes to the entry block too; only the store that arms it
  // stays here, where the loop starts.
  Value pi = loopScratchSlot(builder, wsOp, loc, ptrT, iterTy);
  builder.setInsertionPointToEnd(preBlock);
  if (isChunked) {
    Value more = emitChunkCall(
        plan.firstChunk.empty() ? plan.nextChunk : plan.firstChunk);
    LLVM::CondBrOp::create(builder, loc, more,
      chunkBody, mlir::ValueRange{}, afterBlock, mlir::ValueRange{});

    builder.setInsertionPointToEnd(chunkBody);
    loopStart = fromIdx(LLVM::LoadOp::create(builder, loc, chunkIdxTy, plb));
    loopEnd   = fromIdx(LLVM::LoadOp::create(builder, loc, chunkIdxTy, pub));
    LLVM::StoreOp::create(builder, loc, loopStart, pi);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);
  } else {
    LLVM::StoreOp::create(builder, loc, loopStart, pi);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);
  }

  Block *loopExit = isChunked ? chunkLatch : afterBlock;

  builder.setInsertionPointToEnd(loopHeader);
  Value curI = LLVM::LoadOp::create(builder, loc, iterTy, pi);
  Value cond = LLVM::ICmpOp::create(builder, loc, cmpPred, curI, loopEnd);
  LLVM::CondBrOp::create(builder, loc, cond,
    loopBody, mlir::ValueRange{}, loopExit, mlir::ValueRange{});

  auto &nestRegion = loopNest.getRegion();
  auto &nestFirst  = nestRegion.front();
  nestFirst.getArgument(0).replaceAllUsesWith(curI);
  for (auto &innerOp : llvm::make_early_inc_range(nestRegion.back().getOperations()))
    if (innerOp.getName().getStringRef() == "omp.yield" ||
        innerOp.getName().getStringRef() == "omp.terminator")
      innerOp.erase();

  if (nestRegion.hasOneBlock()) {
    builder.setInsertionPointToEnd(loopBody);
    SmallVector<Operation *> opsToMove;
    for (auto &innerOp : nestFirst.getOperations())
      opsToMove.push_back(&innerOp);
    for (auto *innerOp : opsToMove)
      innerOp->moveBefore(loopBody, loopBody->end());
    builder.setInsertionPointToEnd(loopBody);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopLatch);
  } else {
    // Splice ALL blocks before loopLatch first, then move the first block's ops
    // into loopBody — this preserves branch targets.
    SmallVector<Block *> blocksToMove;
    for (auto &blk : nestRegion)
      if (&blk != &nestFirst) blocksToMove.push_back(&blk);
    for (auto *blk : blocksToMove)
      blk->moveBefore(loopLatch);

    builder.setInsertionPointToEnd(loopBody);
    SmallVector<Operation *> firstOps;
    for (auto &innerOp : nestFirst.getOperations())
      firstOps.push_back(&innerOp);
    for (auto *innerOp : firstOps)
      innerOp->moveBefore(loopBody, loopBody->end());

    builder.setInsertionPointToEnd(blocksToMove.back());
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopLatch);
  }

  builder.setInsertionPointToEnd(loopLatch);
  Value nextI = LLVM::AddOp::create(builder, loc, curI, step);
  LLVM::StoreOp::create(builder, loc, nextI, pi);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);

  if (isChunked) {
    builder.setInsertionPointToEnd(chunkLatch);
    Value more = emitChunkCall(plan.nextChunk);
    LLVM::CondBrOp::create(builder, loc, more,
      chunkBody, mlir::ValueRange{}, afterBlock, mlir::ValueRange{});
  }

  builder.setInsertionPointToStart(afterBlock);
  emitPlanCalls(plan.post, builder);

  wsOp.erase();
  return success();
}

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

    // Step 1: outline the region constructs.
    SmallVector<ConstructOp> constructs;
    module.walk([&](ConstructOp op) {
      if (!op.getBody().empty()) constructs.push_back(op);
    });
    int counter = 0;
    for (auto op : constructs)
      outlineConstruct(op, module, counter);

    // Step 2: lower omp.wsloop ops using the DSL evaluator.
    SmallVector<omp::WsloopOp> wsloops;
    module.walk([&](omp::WsloopOp op) { wsloops.push_back(op); });

    for (auto wsOp : wsloops) {
      if (!wsOp->getBlock()) continue;

      llvm::StringMap<dsl::Value> ctx;
      ctx["ident"]      = dsl::makeStr("%ident");
      ctx["global_tid"] = dsl::makeStr("%gtid");
      ctx["lower"]      = dsl::makeStr("%lb");
      ctx["upper"]      = dsl::makeStr("%ub");
      ctx["step"]       = dsl::makeStr("%step");
      ctx["last"]       = dsl::makeStr("%last");
      // The bounds by value, next to the in/out slots: which upper bound a
      // dispatch API wants is its own convention, so both are offered.
      ctx["lower_val"]  = dsl::makeStr("%lb_val");
      ctx["upper_val"]  = dsl::makeStr("%ub_val");
      ctx["upper_incl"] = dsl::makeStr("%ub_incl");
      // Absent when the clause named none, which is what lets the rules write
      // when has(chunk) and fall back to their own default.
      ctx["chunk"] = wsOp.getScheduleChunk() ? dsl::makeStr("%chunk")
                                             : dsl::makeNull();
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

      // Carry on to the next loop: a rule file with a mistake in it usually has
      // the same mistake in every loop that reads it.
      if (failed(lowerWsloop(wsOp, module, *plan)))
        signalPassFailure();
    }

    // Step 3: hoist allocas.  Last, so it sees both what the front-end left
    // inside a loop and anything the two steps above added.
    module.walk([](func::FuncOp fn) { hoistStaticAllocas(fn); });

    // The trailing team barrier redundant with the fork's join is dropped by
    // OmpBarrierElimPass instead, on the omp dialect, so one rule serves all three.
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::createOmpOutliningPass(std::string dslFile, std::string runtime) {
  return std::make_unique<OmpOutliningPass>(
      std::move(dslFile), std::move(runtime));
}
