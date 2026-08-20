// OmpOutliningPass.cpp
//
// Two responsibilities in one pass:
//
// 1. PARALLEL OUTLINING — for each omp_lower.construct with a body region:
//    - Collect captures, create @outlined_parallel_N func.func
//    - Move region body into it, wire captured values as block args
//    - Emit __kmpc_fork_call (iomp) or GOMP_parallel (libgomp) with captures
//    - Create per-flags __omp_ident_<hex> globals (Clang-parity ident_t) on
//      demand via getOrCreateIdent; the fork path requests the default one
//
//    Capture strategies:
//      by_pointer (iomp): each capture passed as a separate pointer argument
//      packed (libgomp):  all captures packed into an alloca'd struct,
//                         a single ptr to the struct is passed as 'data'
//
// 2. WSLOOP LOWERING — for each omp.wsloop surviving inside outlined funcs:
//    - Extract context (schedule, nowait, bounds) from the omp.loop_nest
//    - Call dsl::Evaluator::buildPlan(runtime, "wsloop", ctx) to get the plan
//    - Emit plan.pre (runtime init call OR `emit thread_bounds` → block chunk),
//      then an explicit loop, then plan.post

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

// Partition captures into the three special-cased kinds shared by every
// outlining path: privatizer captures, scalar-alloca captures, and ptr-alloca
// captures.
// Anything in none of these sets is captured as-is.  The order matters: each
// test excludes values already claimed by an earlier set.
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

// firstprivate snapshot timing: a firstprivate value must be captured *by value*
// so it is snapshotted into the capture struct at construct creation (loaded at
// the call site by storeCapturesToBase), not read through a captured pointer at
// entry.  Otherwise a deferred task that runs after the source was mutated (the
// canonical firstprivate(i) spawn loop) observes the wrong value.  The sources
// are the leading captures (injected first by OmpToOmpLowerPass, one per
// privatizer block arg); classifyCaptures leaves them as plain captures because
// their first in-region use is the injected marker cast, not a load.  Force
// scalar-alloca sources into the by-value bucket here.  Non-alloca (by-pointer)
// sources can't be packed by value this way and keep the read-at-entry
// behaviour.  Harmless for `parallel` (creation coincides with the fork).
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

// Clause operands are carried on the ConstructOp as a single variadic list;
// the 1:1 clause_names attribute says which clause each entry belongs to
// ("num_threads", "if_clause", ...).  Returns the operand for `name`, or null.
static Value getClauseOperand(ConstructOp op, llvm::StringRef name) {
  auto names = op.getClauseNames();
  if (!names) return Value();
  for (auto [i, n] : llvm::enumerate(*names))
    if (llvm::cast<StringAttr>(n).getValue() == name)
      return op.getClauseOperands()[i];
  return Value();
}

// Normalise a clause value to i1 for use as an llvm.cond_br condition /
// arith.select predicate (if-clause values are typically already i1).
static std::string getPropStr(ConstructOp op, llvm::StringRef key) {
  auto dict = op.getPropDict();
  if (!dict) return "";
  if (auto sa = llvm::dyn_cast_or_null<StringAttr>(dict.get(key)))
    return sa.getValue().str();
  return "";
}

// capture_strategy is the single ABI discriminator: the delivery mechanism it
// names uniquely entails the outlined-function signature.
//   - by_pointer -> microtask   void(gtid, btid, cap0, cap1, ...)
//   - packed     -> closure      void(ptr data)   (captures in one struct)
//   - shareds    -> task routine i32(gtid, ptr task), captures via
//                   task->shareds, emitted by outlineTaskEntry.
enum class CaptureAbi { ByPointer, Packed, Shareds };

static std::optional<CaptureAbi> parseCaptureAbi(llvm::StringRef s) {
  if (s == "by_pointer") return CaptureAbi::ByPointer;
  if (s == "packed")     return CaptureAbi::Packed;
  if (s == "shareds")    return CaptureAbi::Shareds;
  return std::nullopt;
}

// Read a DSL-owned struct-layout property into an LLVM literal struct type.
// The expansion itself lives in PlanEmit: the plan pass reads the same layout.
static LLVM::LLVMStructType getPropStructType(ConstructOp op, llvm::StringRef key,
    MLIRContext *ctx, LLVM::LLVMStructType fallback) {
  return parseStructProp(ctx, getPropStr(op, key), fallback);
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

// ---------------------------------------------------------------------------
// Capture-layout helpers (shared by closure and task_entry paths)
// ---------------------------------------------------------------------------
// Both paths use the "packed" capture topology: the SAME struct
// { field_0, ..., field_N-1 }.  They differ only in where it lives and how the
// body reaches it:
//   closure (libgomp) : struct on the caller stack; body gets a direct ptr.
//   task_entry (iomp) : runtime-allocated; body reaches it via
//                       load(task->shareds).
// These three helpers factor out the field-type layout, the prolog unpack, and
// the call-site populate so the two paths share one implementation.

// Build the capture-struct field types (scalar-alloca capture → its element
// type, packed by value; otherwise the captured value's own type) and return
// the literal struct type.  `fieldTypes` is filled with the field types.
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

// In the function prolog (builder `b`), unpack each capture from the struct at
// `structBase`, rewrite its uses inside `fnBody`, and — if `loadedCaptures` is
// non-null — record the per-capture unpacked value (used by the packed path's
// firstprivate handling).  `one64` is a shared i64 constant 1 for the allocas.
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
      // Private capture (loop IV etc.): fresh per-thread alloca, not read from
      // the struct — each thread gets an independent copy.
      auto srcAlloca = captures[i].getDefiningOp<LLVM::AllocaOp>();
      result = LLVM::AllocaOp::create(b, loc, ptr, srcAlloca.getElemType(),
        one64);
      replaceUsesInRegion(fnBody, captures[i], result);
    } else if (scalarAllocaCaptures.contains(captures[i])) {
      // Scalar packed by value: load the scalar from the struct, then stash it
      // in a fresh alloca so existing load/store-of-alloca patterns keep working
      // (mem2reg/SROA promote the single-store alloca to a register).
      Type elemTy = captures[i].getDefiningOp<LLVM::AllocaOp>().getElemType();
      Value gep = LLVM::GEPOp::create(b, loc, ptr, structTy, structBase,
        ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
      Value scalar = LLVM::LoadOp::create(b, loc, elemTy, gep);
      result = LLVM::AllocaOp::create(b, loc, ptr, elemTy, one64);
      LLVM::StoreOp::create(b, loc, scalar, result);
      replaceUsesInRegion(fnBody, captures[i], result);
    } else if (ptrAllocaCaptures.contains(captures[i])) {
      // Pointer packed by value: same idea, the field holds the pointer.
      Value gep = LLVM::GEPOp::create(b, loc, ptr, structTy, structBase,
        ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
      Value pv = LLVM::LoadOp::create(b, loc, ptr, gep);
      result = LLVM::AllocaOp::create(b, loc, ptr, ptr, one64);
      LLVM::StoreOp::create(b, loc, pv, result);
      replaceUsesInRegion(fnBody, captures[i], result);
    } else {
      // Plain capture: load the value from the struct.
      Value gep = LLVM::GEPOp::create(b, loc, ptr, structTy, structBase,
        ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
      result = LLVM::LoadOp::create(b, loc, captures[i].getType(), gep);
      replaceUsesInRegion(fnBody, captures[i], result);
    }
    if (loadedCaptures) loadedCaptures->push_back(result);
  }
}

// At the call site (builder `b`), compute the value that goes into each capture
// field.  Mirrors unpackCapturesFromBase: a private slot is left undef, a
// scalar/ptr alloca is loaded by value, a plain capture is passed as-is.
//
// Separate from the store below because the two halves can happen in different
// passes.  The iomp task writes its captures into a block the *plan* pass
// allocates (task->shareds), and only this pass knows the classification, so it
// resolves the values here and hands them over as bindings; the plan pass then
// only has to GEP and store them in order.
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

// Resolve the captures and store them into the struct at `structBase`, for the
// paths that build the struct themselves (the packed/closure call site).
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

// ---------------------------------------------------------------------------
// Outlined-body prep helpers (shared by the parallel/closure and task_entry
// paths).  The capture-unpacking prolog differs per ABI and stays in each
// caller; these factor out the identical head (strip private_vars + takeBody)
// and tail (dead-cast cleanup + terminator rewrite).
// ---------------------------------------------------------------------------

// Strip omp private_vars from the region, move it into outlinedFn, and return
// the entry block.  preexistingArgs is filled with the block args that arrived
// with the region (privatizer args) so the caller can wire or drop them.
static Block &takeOutlinedBody(Region &body, func::FuncOp outlinedFn,
                               SmallVectorImpl<BlockArgument> &preexistingArgs) {
  // Strip omp.wsloop/omp.parallel private_vars operands before takeBody so
  // replaceUsesInRegion cannot corrupt those operand references.
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

// Erase injected unrealized_conversion_cast marker ops that have no users.
static void eraseDeadCasts(func::FuncOp outlinedFn) {
  SmallVector<Operation *> casts;
  outlinedFn.getBody().walk([&](UnrealizedConversionCastOp c) {
    if (c->use_empty()) casts.push_back(c);
  });
  for (auto *c : casts) c->erase();
}

// Replace every omp.terminator with a func.return.  emitReturn builds the
// return op at the terminator's location (void for closure/microtask, an i32
// zero for task_entry), once per terminator so each gets its own operands.
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

// ---------------------------------------------------------------------------
// 1a. IOMP TASK OUTLINING  (capture_strategy = shareds)
// ---------------------------------------------------------------------------
// Outlines an omp_lower.construct for an iomp task into an entry function
// i32(i32 gtid, ptr task) that loads task->shareds and unpacks the captures
// from it, then hands the construct to PlanLoweringPass with the entry pointer,
// the two ABI sizes and the resolved capture values bound to it.  The call
// sequence itself — alloc, the write into task->shareds, and the if0 branch —
// is stated in rules.dsl and emitted there.
// An omp.taskwait nested in the task body is lowered by the plan pass like any
// other leaf; %gtid is bound here to the entry's arg 0 (an i32 value, unlike
// the microtask ptr-to-i32 convention).
// Supports implicit captures and explicit firstprivate clauses (copy-in into a
// task-private slot, mirroring the packed path).  v1 limitations: pure
// `private` clauses are not wired (diagnosed, not miscompiled); no final
// clause; task_flags = 1 (tied); depend/nowait on taskwait ignored;
// omp.barrier is illegal in a task region and diagnosed.
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

  // Collect + classify captures (same scheme as the packed strategy).
  SmallVector<Value> captures = collectCaptures(body);
  llvm::SetVector<Value> privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures;
  classifyCaptures(captures, body, privateCaptures, scalarAllocaCaptures,
                   ptrAllocaCaptures);

  // firstprivate values are snapshotted at task creation (see helper).
  forceFirstprivateByValue(body, captures, privateCaptures,
                           scalarAllocaCaptures, ptrAllocaCaptures);

  // shareds struct: one field per capture (element type for a scalar-alloca
  // capture, else the captured value's own type).
  SmallVector<Type> fieldTypes;
  auto sharedsTy =
      buildCaptureStruct(ctx, captures, scalarAllocaCaptures, fieldTypes);

  // kmp_task_t header: { void *shareds, routine, kmp_int32 part_id, data1,
  // data2 } — 40 bytes on 64-bit; only field 0 (shareds) is accessed here, but
  // the full size is passed to __kmpc_omp_task_alloc so the runtime's writes to
  // the header stay in bounds.  The layout is DSL-owned (`kmp_task_t = struct
  // (...)` in the task construct, like `task_flags`); the literal below is the
  // fallback for DSL files that don't declare it.
  auto kmpTaskTy = getPropStructType(op, "kmp_task_t", ctx,
      LLVM::LLVMStructType::getLiteral(ctx, {ptr, ptr, i32t, ptr, ptr}));

  std::string fnName =
      "outlined_" + op.getConstructName().str() + "_" + std::to_string(counter++);

  // Build the entry function  i32(i32 gtid, ptr task).
  OpBuilder builder(ctx);
  if (auto parentFn = op->getParentOfType<func::FuncOp>())
    builder.setInsertionPoint(parentFn);
  else
    builder.setInsertionPointToStart(module.getBody());

  auto outlinedFn = func::FuncOp::create(loc, fnName,
    FunctionType::get(ctx, {i32t, ptr}, {i32t}));
  outlinedFn.setVisibility(SymbolTable::Visibility::Nested);
  builder.insert(outlinedFn);

  // Move the body into the entry function.  staleArgs receives any privatizer
  // block args that arrived with the region.  A firstprivate arg gets a
  // task-private copy below (its uses are rewritten); the leftover (now-dead)
  // args are dropped after capture unpacking so the entry keeps its
  // i32(i32 gtid, ptr task) ABI.
  SmallVector<BlockArgument> staleArgs;
  Block &entry = takeOutlinedBody(body, outlinedFn, staleArgs);

  // Prepend the entry args: [gtid, task, ...stale privatizer args].
  entry.insertArgument(0u, i32t, loc);                         // gtid
  BlockArgument taskArg = entry.insertArgument(1u, ptr, loc);  // task

  // Prolog: shareds = load &task->shareds, then unpack each capture from it.
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

    // firstprivate copy-in — same scheme as the packed/closure path.  The
    // privatizer source ptrs were injected at the start of the region by
    // OmpToOmpLowerPass, so collectCaptures placed them first: loadedCaptures[i]
    // is the source ptr for staleArgs[i].  For each, allocate a task-private
    // slot, copy *source into it, and rewrite the block arg's uses to it.
    // (Shares the packed path's assumption that the privatizer args line up 1:1
    // with the leading captures, which holds for firstprivate-only clauses.)
    for (size_t i = 0; i < staleArgs.size() && i < loadedCaptures.size(); i++) {
      BlockArgument dstArg = staleArgs[i];
      Value srcPtr = loadedCaptures[i];
      if (!llvm::isa<LLVM::LLVMPointerType>(srcPtr.getType())) continue;
      // Element type inferred from a load of the block arg inside the body.
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

  // Drop privatizer block args now made dead by the copy-in.  Any that survive
  // with live uses are an unsupported clause shape (a pure `private` clause, or
  // a firstprivate whose element type could not be inferred): diagnose it rather
  // than silently emit a wrong-ABI entry (trailing params the runtime never
  // passes).
  for (auto arg : llvm::reverse(staleArgs)) {
    if (arg.use_empty())
      entry.eraseArgument(arg.getArgNumber());
    else
      op.emitError("omp-outline: iomp task has an unsupported private/"
                   "firstprivate clause; entry ABI would break");
  }

  // Remove injected privatizer marker casts.
  eraseDeadCasts(outlinedFn);

  // Finalize the function type to match the entry block.
  outlinedFn.setFunctionType(
      FunctionType::get(ctx, entry.getArgumentTypes(), {i32t}));

  // Replace omp.terminator with `func.return %0 : i32`.
  replaceTerminatorsWithReturn(outlinedFn, [&](OpBuilder &tb, Location l) {
    Value zero = LLVM::ConstantOp::create(tb, l, i32t,
      IntegerAttr::get(i32t, 0));
    func::ReturnOp::create(tb, l, ValueRange{zero});
  });

  // A barrier is illegal inside a task region in OpenMP.  Diagnose it before
  // binding below, so it is reported rather than lowered.
  outlinedFn.walk([&](ConstructOp inner) {
    if (inner.getConstructName() == "barrier")
      inner->emitError(
          "omp-outline: 'omp.barrier' is not valid inside a task region");
  });

  // A taskwait nested directly in the task body — `parallel { task { taskwait } }`
  // — is lowered by PlanLoweringPass like any other leaf.  Bind the thread id it
  // cannot derive: the shareds entry ABI is i32(i32 gtid, ptr task), so %gtid is
  // arg 0 *by value*, not the microtask's ptr-to-i32.
  bindGtidOnLeaves(outlinedFn, ctx, [&](OpBuilder &, Location) -> Value {
    return outlinedFn.getBody().front().getArgument(0);
  });

  // ---- Call site ----
  // Hand the construct over to PlanLoweringPass, like every other region
  // construct.  This pass attaches only what it alone can produce — the entry
  // pointer, the two ABI sizes and the resolved capture values — and leaves the
  // construct standing.  Everything else is stated in rules.dsl and emitted
  // there: the alloc call and the binding of its result, the write into
  // task->shareds, and the if0 branch.
  //
  // ident and the global thread id are not bound: the plan pass materialises
  // both on first reference, as it does for barrier and taskwait.
  builder.setInsertionPoint(op);

  DataLayout dl(module);
  uint64_t taskSize    = dl.getTypeSize(kmpTaskTy).getFixedValue();
  uint64_t sharedsSize = dl.getTypeSize(sharedsTy).getFixedValue();

  Value fnPtr = func::ConstantOp::create(builder, loc,
    outlinedFn.getFunctionType(), FlatSymbolRefAttr::get(ctx, fnName));
  Value fnPtrCast = UnrealizedConversionCastOp::create(builder, loc,
    TypeRange{ptr}, ValueRange{fnPtr}).getResult(0);

  // The values the shareds fields receive.  Resolved here because the
  // private / scalar-alloca / ptr-alloca classification is outlining knowledge;
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
  // The entry address, under the one name the task plan uses for it: as the
  // alloc call's `body` argument, and as the callee of the direct call on the
  // undeferred side of `if`.
  bind("body", fnPtrCast);
  // __kmpc_omp_task_alloc's two size arguments: the kmp_task_t header (so the
  // runtime's writes to it stay in bounds) and the shareds block.
  bind("task_size", LLVM::ConstantOp::create(builder, loc, i64t,
                      IntegerAttr::get(i64t, (int64_t)taskSize)));
  bind("shareds_size", LLVM::ConstantOp::create(builder, loc, i64t,
                         IntegerAttr::get(i64t, (int64_t)sharedsSize)));
  // captures is a list: every operand under this name belongs to it, and
  // list_names keeps it a list even when there are none.
  for (Value v : capVals) bind("%captures", v);

  op->setOperands(operands);
  op.setClauseNamesAttr(ArrayAttr::get(ctx, names));
  op.setListNamesAttr(ArrayAttr::get(ctx,
      {StringAttr::get(ctx, "%captures")}));
  warnIgnoredClauses(op);
  return;   // deliberately not erased: the plan pass consumes it
}

// Lower leaf omp ops (omp.barrier, omp.taskwait) that ride inside an outlined
// function by applying a precomputed DSL plan.  These constructs have no body
// and no captures; the plan's invoke calls use only the symbolic %ident/%gtid
// args.  ident resolves the same everywhere, but the gtid source depends on the
// enclosing function's ABI, so the caller supplies it via resolveGtid:
//   - microtask entry  void(ptr gtid, ptr btid, ...):  load arg 0 (ptr to i32);
//   - task entry       i32(i32 gtid, ptr task):        arg 0 is the gtid value.
// Bind %gtid on every leaf ConstructOp inside an outlined function.
//
// A barrier or taskwait nested in a parallel/task rides into the outlined body
// as an empty-body ConstructOp carrying its own plan.  PlanLoweringPass emits
// the calls; the only thing it cannot derive is the thread id, which comes from
// the signature outlining just created.  `makeGtid` produces it (a load of the
// microtask's arg 0, or the task entry's by-value arg), and it is materialised
// once at the top of the entry block so it dominates every leaf below.
//
// The binding travels as a regular operand plus its name in clause_names, the
// same channel the clause values use.
// Does any action in this plan block name the given token?  Recurses into
// branch arms, and looks at a branch's condition as well as call arguments.
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

// Does the construct's plan name the token anywhere (pre, invoke or post)?
static bool planNamesToken(ConstructOp op, llvm::StringRef token) {
  return blockNamesToken(op.getPre(), token) ||
         blockNamesToken(op.getInvoke(), token) ||
         blockNamesToken(op.getPost(), token);
}

// A clause the construct carries that no plan action names is a clause this
// runtime's rules have no lowering for.  Emitting the construct anyway would
// change semantics without saying so — if(false) must run the region
// serialized, num_threads must size the team, proc_bind asks for an affinity
// policy — so each one gets a warning.  The check is per construct, not per
// runtime: the same clause can be lowered by one construct and dropped by
// another, and only the plan knows which.
static void warnIgnoredClauses(ConstructOp op) {
  // The tokens a plan can name the clause by, and how OpenMP spells it.  Most
  // have one; proc_bind has two — the push argument (iomp) and the flags word
  // GOMP always takes — and naming either one is a lowering.
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
  // Only leaves whose plan actually names %gtid: binding the rest would
  // materialise a thread id nothing reads, and on the closure runtimes
  // (libgomp, pmsis) that is a dead undef at the top of every outlined
  // function — GOMP_barrier and friends take no arguments.
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


// ---------------------------------------------------------------------------
// 1. PARALLEL OUTLINING
// ---------------------------------------------------------------------------

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

  // iomp task uses a distinct ABI — the shareds signature: an
  // i32(i32 gtid, ptr task) entry whose captures live in a runtime-allocated
  // shareds block, emitted via the __kmpc_omp_task_alloc/task two-call
  // sequence.
  if (abi == CaptureAbi::Shareds) {
    outlineTaskEntry(op, module, counter);
    return;
  }

  // Collect all values used inside the region but defined outside.
  SmallVector<Value> captures = collectCaptures(body);

  // Classify captures before takeBody (the body region is consumed after).
  // - private captures: privatizer-driven;
  // - scalar-alloca captures: allocas holding a scalar (int/float) whose first
  //   region use is a load.  For the packed/closure strategy these are stored
  //   by value in the capture struct so the outlined function receives the
  //   scalar directly — eliminating the double indirection (struct field →
  //   alloca pointer → scalar) that would otherwise appear on every use inside
  //   the innermost loop and block LLVM's LICM from hoisting those loads;
  // - ptr-alloca captures: allocas holding a pointer (e.g. array ptr) whose
  //   first region use is a load.  The pointer VALUE is packed directly into
  //   the struct field instead of the alloca address, saving one dereference
  //   per array access inside the outlined function.
  llvm::SetVector<Value> privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures;
  classifyCaptures(captures, body, privateCaptures, scalarAllocaCaptures,
                   ptrAllocaCaptures);

  // Snapshot firstprivate values at creation for the packed strategy (the
  // by-value capture struct).  Only the packed path consumes these buckets; the
  // by_pointer path passes captures as individual args.  See helper.
  if (abi == CaptureAbi::Packed)
    forceFirstprivateByValue(body, captures, privateCaptures,
                             scalarAllocaCaptures, ptrAllocaCaptures);

  // Name the outlined function after the construct ("outlined_parallel_N",
  // "outlined_task_N", ...).  Parallel keeps its historical name.
  std::string fnName =
      "outlined_" + op.getConstructName().str() + "_" + std::to_string(counter++);

  // Build outlined function argument types.
  SmallVector<Type> fnArgTypes;
  if (abi == CaptureAbi::ByPointer) {
    // iomp microtask: void(ptr gtid, ptr btid, cap0, cap1, ...)
    fnArgTypes.push_back(ptrTy(ctx)); // ptr gtid
    fnArgTypes.push_back(ptrTy(ctx)); // ptr btid
    for (auto cap : captures) fnArgTypes.push_back(cap.getType());
  } else {
    // packed/closure: void(ptr data) — data points to the capture struct.
    fnArgTypes.push_back(ptrTy(ctx));
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

  // Move the body into the outlined function.  privatizerArgs receives any
  // privatizer block args that arrived with the region (from omp.parallel),
  // saved before inserting capture args.
  SmallVector<BlockArgument> privatizerArgs;
  Block &entry = takeOutlinedBody(body, outlinedFn, privatizerArgs);

  if (abi == CaptureAbi::Packed) {
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
    // Insert the data ptr at position 0.  The entry block may already have
    // privatizer args — we insert before them.
    entry.insertArgument(0u, ptrTy(ctx), loc);
    BlockArgument dataPtr = entry.getArgument(0u);

    // Build the capture struct type: { cap_0_type, cap_1_type, ... }
    SmallVector<Type> fieldTypes;
    auto structTy =
        buildCaptureStruct(ctx, captures, scalarAllocaCaptures, fieldTypes);

    // In the function prolog, unpack each capture from the struct.
    if (!captures.empty()) {
      OpBuilder prologue(&entry, entry.begin());
      Value one64 = LLVM::ConstantOp::create(prologue, loc,
        IntegerType::get(ctx, 64),
        IntegerAttr::get(IntegerType::get(ctx, 64), 1));
      // Keep loaded values so privatizer handling can reuse them.
      SmallVector<Value> loadedCaptures;
      unpackCapturesFromBase(prologue, loc, one64, dataPtr, structTy, captures,
        privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures,
        outlinedFn.getBody(), &loadedCaptures);
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
  } else if (abi == CaptureAbi::ByPointer) {
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
  }

  // Remove privatizer block args now made dead by the copy-in.  They were
  // replaced by private copies in the prolog, so keeping them would add extra
  // parameters the call site never fills.  A survivor with live uses is an
  // unsupported clause shape (a pure `private` clause, a non-scalar/aggregate
  // firstprivate, or one whose element type couldn't be inferred): diagnose it
  // rather than silently emit a wrong-ABI outlined function.  Mirrors the iomp
  // path (outlineTaskEntry).
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

  // Remove injected unrealized_conversion_cast marker ops (no users).
  eraseDeadCasts(outlinedFn);

  // Erase unused capture args from the entry block and filter captures list.
  // This removes captures that were only needed as privatizer sources
  // (e.g., source allocas for private vars that don't need copying).
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

  // Update function type to match actual entry block args.
  SmallVector<Type> finalArgTypes;
  for (auto arg : entry.getArguments())
    finalArgTypes.push_back(arg.getType());
  outlinedFn.setFunctionType(FunctionType::get(ctx, finalArgTypes, {}));

  // Mark capture pointer args `noalias` so the optimiser can hoist the base-
  // pointer loads out of the outlined loop and vectorise the body.  Each
  // capture arrives as a pointer to a distinct caller-side slot (one spilled
  // slot per shared variable, materialised by OmpToOmpLowerPass), so a store
  // through one capture never touches the storage of another — the assertion
  // is sound.  Without it the by_pointer microtask body stays scalar: a store
  // through one capture may, for all the optimiser knows, clobber the pointer
  // reached through another, which blocks LICM and vectorisation.  gtid/btid
  // are runtime-owned and left untouched (captures start at index 2).
  if (abi == CaptureAbi::ByPointer) {
    UnitAttr noalias = UnitAttr::get(ctx);
    for (unsigned i = 2, e = entry.getNumArguments(); i < e; i++)
      if (llvm::isa<LLVM::LLVMPointerType>(entry.getArgument(i).getType()))
        outlinedFn.setArgAttr(i, "llvm.noalias", noalias);
  }

  // Replace omp.terminator with func.return (direct blocks only).
  replaceTerminatorsWithReturn(outlinedFn, [](OpBuilder &tb, Location l) {
    func::ReturnOp::create(tb, l);
  });

  // Leaf constructs (barrier, taskwait) that rode into the outlined function
  // carry their own plan and are lowered by PlanLoweringPass.  The one thing it
  // cannot know is the thread id: this is the microtask entry, so %gtid is a
  // load of arg 0 (a ptr to i32 gtid), which only exists because of outlining.
  // Bind it here, materialised once in the entry block so it dominates every
  // leaf in the function.
  bindGtidOnLeaves(outlinedFn, ctx, [&](OpBuilder &bb, Location bloc) -> Value {
    auto &fnEntry = outlinedFn.getBody().front();
    if (fnEntry.getNumArguments() >= 2)
      return LLVM::LoadOp::create(bb, bloc, i32Ty(ctx), fnEntry.getArgument(0));
    return LLVM::UndefOp::create(bb, bloc, i32Ty(ctx));
  });

  // ---- Hand the construct to PlanLoweringPass ----
  // This pass emits no runtime call.  It attaches what only it can produce —
  // the outlined function's address, the capture struct or the individual
  // capture values, the ABI sizes — as named operands and leaves the construct
  // standing.  Everything the DSL states in pre/invoke/post is emitted there.
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

  // The outlined function's address, under every spelling a plan may use for
  // it: the evaluation context seeds `body` with "outlined_parallel" for
  // parallel and with "body" for task, and a plan can also name it as a callee
  // (the direct microtask call on the serialized side of `if`).  Missing one
  // resolves to an undef pointer and forks into nothing.
  Value fnPtr = func::ConstantOp::create(builder, loc,
    outlinedFn.getFunctionType(), FlatSymbolRefAttr::get(ctx, fnName));
  Value fnPtrCast = UnrealizedConversionCastOp::create(
    builder, loc, TypeRange{ptrTy(ctx)}, ValueRange{fnPtr}).getResult(0);
  bind("body", fnPtrCast);
  bind("outlined_parallel", fnPtrCast);
  bind("outlined_task", fnPtrCast);

  // proc_bind names a compile-time affinity policy, so unlike num_threads it
  // arrives as an attribute and the constant for it has to be made here rather
  // than resolved from an operand.  Two tokens carry the one value: `proc_bind`
  // is the clause, bound only when one was given, and `proc_bind_flags` is the
  // flags word GOMP always takes, 0 when no policy was asked for.  Each is
  // bound only if the plan names it — elsewhere it would be a constant nothing
  // reads, and that case is what the ignored-clause warning covers instead.
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
    // -------------------------------------------------------------------------
    // PACKED / CLOSURE: one capture struct on the stack, handed over by pointer
    // -------------------------------------------------------------------------
    // Scalar/ptr-alloca captures are packed by value (see buildCaptureStruct /
    // storeCapturesToBase); the struct type also gives env_size / env_align.
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

    // The struct's layout: GOMP_task takes it as `long arg_size, long
    // arg_align` (libgomp memcpys arg_size bytes into task-private memory when
    // cpyfn is NULL).  Computed via the module DataLayout; alignment falls back
    // to 16, a valid power-of-2 >= any field's own alignment.
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
    // -------------------------------------------------------------------------
    // BY_POINTER (iomp microtask): each capture is its own trailing argument
    // -------------------------------------------------------------------------
    // A private capture (a loop IV) gets a fresh local alloca so each thread has
    // its own copy rather than sharing the caller's.
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

    // The serialized side of `if` calls the microtask directly, and the
    // microtask ABI takes gtid and btid *by pointer* — two slots only this pass
    // can make.  Emitted just for that path, so a parallel with no if clause
    // gets neither, nor the thread id they need.
    if (getClauseOperand(op, "if_clause")) {
      std::string gtidFnName = getPropStr(op, "global_tid_function");
      Value gtid;
      if (gtidFnName.empty()) {
        // No gtid source under this runtime; declaring a nameless function
        // would fail the verifier.
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
      // Hand the thread id over too: the serialized calls need it, and the
      // store above already had to materialise it.  Without this the plan pass
      // would emit a second __kmpc_global_thread_num of its own.
      bind("%gtid", gtid);
    }

    // captures is a list: every operand under this name belongs to it, and
    // list_names keeps it a list even when there are none.
    for (Value v : capVals) bind("%captures", v);
    op.setListNamesAttr(ArrayAttr::get(ctx,
        {StringAttr::get(ctx, "%captures")}));
  }

  op->setOperands(operands);
  op.setClauseNamesAttr(ArrayAttr::get(ctx, names));

  warnIgnoredClauses(op);
}

// ---------------------------------------------------------------------------
// 2. WSLOOP LOWERING — driven by DSL plan
// ---------------------------------------------------------------------------

// Returns failure once a diagnostic has been emitted, so the caller can fail
// the pass: a wsloop this cannot lower is left standing in the output, and a
// module still holding an omp.wsloop is not something the rest of the pipeline
// can make sense of.
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

  Value lb = lbs[0], ub = ubs[0], step = steps[0];
  Type iterTy = lb.getType();

  // omp.loop_nest bounds may be inclusive or exclusive depending on how the
  // front-end emitted them (the "inclusive" keyword on the op means the upper
  // bound is the last valid iteration value, not one-past-the-end).
  // Normalise to exclusive here so all downstream trip-count and range
  // arithmetic is uniform: exclusive_ub = inclusive_ub + step.
  if (loopNest.getLoopInclusive())
    ub = LLVM::AddOp::create(builder, loc, ub, step);

  // Get gtid from enclosing outlined function. (idents are resolved per call
  // argument below, since init/fini and the trailing barrier use different
  // flags.)
  // For iomp microtask: arg0 = ptr gtid, arg1 = ptr btid → load i32 from arg0.
  // For libgomp closure: arg0 = ptr data (capture struct) — no gtid ptr.
  //   Use omp_get_thread_num() to obtain the current thread id.
  //
  // Only when a call in the plan asks for it.  On the closure runtimes the
  // thread id costs a call to omp_get_thread_num, and an external call is one
  // thing LLVM cannot delete for being unused — so a wsloop whose rules never
  // name %gtid (every libgomp and pmsis one: their loop APIs take no thread id)
  // would carry it into the binary for nothing.
  auto actionsName = [](const std::vector<dsl::PlanAction> &actions,
                        llvm::StringRef token) {
    for (auto &a : actions)
      if (auto *ca = std::get_if<dsl::PlanCall>(&a))
        for (auto &av : ca->args)
          if (auto *sv = std::get_if<dsl::StrVal>(&av))
            if (sv->value == token) return true;
    return false;
  };
  bool needsGtid = actionsName(plan.pre, "%gtid") ||
                   actionsName(plan.invoke, "%gtid") ||
                   actionsName(plan.post, "%gtid") ||
                   actionsName(plan.firstChunk, "%gtid") ||
                   actionsName(plan.nextChunk, "%gtid");

  Value gtidVal;
  if (needsGtid) gtidVal = LLVM::UndefOp::create(builder, loc, iterTy);
  if (auto parentFn = wsOp->getParentOfType<func::FuncOp>(); parentFn && needsGtid) {
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

  // Read DSL properties that drive loop lowering strategy.
  auto getStrProp = [&](llvm::StringRef key) -> std::string {
    auto it = plan.properties.find(key.str());
    if (it == plan.properties.end()) return "";
    if (auto *sv = std::get_if<dsl::StrVal>(&it->second)) return sv->value;
    return "";
  };

  // A property that is present but unreadable is a typo in the rules, and each
  // of these three fails a different way if it quietly falls back: the wrong
  // slot width, the wrong truthiness test, or — worst — an inner loop running
  // one iteration past the end of every chunk.  None of that is worth
  // discovering at run time, so an unusable value is an error and not a
  // default.  Absent still means "use the pass's own", which is what keeps the
  // runtimes that agree with it from declaring anything.
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

  // The index type of this runtime's loop ABI: the width of the bound slots it
  // writes and of the values it takes by value.  It is not always the loop's
  // own: libgomp's GOMP_loop_* family is long-based whatever the induction
  // variable is, while iomp's __kmpc_*_4 matches an i32 one.  Defaulting to the
  // induction variable's type is what keeps every runtime that agrees with it
  // free of conversions — and of any change to the IR it emitted before.
  Type chunkIdxTy = typeProp("chunk_index", iterTy);

  // A construct that declares a next_chunk block is one whose iterations the
  // runtime hands out a chunk at a time: the loop below gets an outer loop
  // around it, asking for a chunk per turn.  The schedule kind is never read
  // here — which schedules are chunked is a statement rules.dsl makes.
  bool isChunked = !plan.nextChunk.empty();
  // What such a call returns (0 = no work left).  iomp's dispatch_next returns
  // an int, GOMP's a C _Bool.
  Type chunkResTy = typeProp("chunk_result", i32Ty(ctx));

  // Whether the upper bound the runtime writes into the slot is the last valid
  // iteration or the one past it.  Read here rather than where the predicate is
  // chosen so a misspelling is caught with the other two.
  std::string chunkBound = getStrProp("chunk_bound");
  if (!chunkBound.empty() && chunkBound != "inclusive" &&
      chunkBound != "exclusive") {
    wsOp.emitError("omp-outline: `chunk_bound = ")
        << chunkBound << "` is neither `inclusive` nor `exclusive`";
    badProp = true;
  }

  // The chunk size rides in from the clause, so unlike everything else in the
  // index vocabulary its type is the input's to choose.  Anything but an
  // integer has no conversion into the ABI's index type, and reaching the
  // conversion below with one would abort rather than diagnose.
  Value chunkVal = wsOp.getScheduleChunk();
  if (chunkVal && !llvm::isa<IntegerType>(chunkVal.getType())) {
    wsOp.emitError("omp-outline: the schedule chunk must be an integer, got ")
        << chunkVal.getType();
    badProp = true;
  }

  // A construct with a first_chunk block and no next_chunk has no way to ask
  // for a second chunk, and isChunked is false, so the block would simply not
  // be emitted: the work-share never registered and every thread running the
  // whole iteration space.  Wrong code, quietly — so refuse it.
  if (!plan.firstChunk.empty() && plan.nextChunk.empty()) {
    wsOp.emitError("omp-outline: this wsloop declares `first_chunk` but no "
                   "`next_chunk`; there is no call to ask for another chunk");
    badProp = true;
  }

  if (badProp) return failure();

  // Convert into and out of that ABI.  Loop indices are signed, so the widening
  // direction is a sign extension.  Both sides are integers by the checks
  // above; a non-integer would leave the value alone and be caught by the
  // verifier rather than crash here.
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

  // Allocate plb, pub, pstride, plast.
  // plb/pub/plast are in/out: initialized here, overwritten by the runtime.
  // pstride is pure output: NOT initialized — runtime writes the stride value.
  Value plb     = LLVM::AllocaOp::create(builder, loc, ptrT, chunkIdxTy, one64);
  Value pub     = LLVM::AllocaOp::create(builder, loc, ptrT, chunkIdxTy, one64);
  Value pstride = LLVM::AllocaOp::create(builder, loc, ptrT, chunkIdxTy, one64);
  Value plast   = LLVM::AllocaOp::create(builder, loc, ptrT, chunkIdxTy, one64);

  // Convert exclusive upper bound to inclusive last-iteration index.
  // omp.loop_nest uses exclusive ub: trip = (ub - lb) / step iterations.
  // __kmpc_for_static_init_4 expects inclusive upper bound: ub_incl = lb + (trip-1)*step
  //   = lb + ((ub-lb)/step - 1)*step  = ub - step
  Value ubInclusive = LLVM::SubOp::create(builder, loc, ub, step);

  // For a chunked construct the slots are pure output — the bounds go in by
  // value through the acquisition call, and every read of a slot happens after
  // the runtime has written it — so there is nothing to seed.
  if (!isChunked) {
    LLVM::StoreOp::create(builder, loc, toIdx(lb),          plb);
    LLVM::StoreOp::create(builder, loc, toIdx(ubInclusive), pub);
    // pstride: no initialization — pure output parameter
    LLVM::StoreOp::create(builder, loc, toIdx(zero32),      plast);
  }

  // Map symbolic DSL names to SSA values.  The loop-specific slots (%lb, %ub,
  // %step, %stride, %last) are the site bindings; ident and %gtid come from the
  // shared vocabulary (see resolveSymbolToken).
  llvm::StringMap<Value> wsBindings;
  wsBindings["%lb"]     = plb;      // in/out lower-bound slot
  wsBindings["%ub"]     = pub;      // in/out upper-bound slot
  wsBindings["%step"]   = step;     // actual loop step
  wsBindings["%stride"] = pstride;  // output ptr for runtime stride
  wsBindings["%last"]   = plast;    // in/out last-iteration flag
  // The bounds by value, for the dispatch APIs that take them that way rather
  // than through the slots above.  Which of the two upper bounds a runtime
  // wants is its own ABI: iomp's dispatch_init is given the last valid
  // iteration, libgomp's loop_start the one-past-the-end.
  wsBindings["%lb_val"]  = lb;
  wsBindings["%ub_val"]  = ub;           // exclusive, as loop_nest was normalised
  wsBindings["%ub_incl"] = ubInclusive;  // last valid iteration
  if (chunkVal) wsBindings["%chunk"] = chunkVal;  // integer, checked above

  // Everything naming a loop index crosses into the runtime's index type: the
  // bounds, the step, the chunk size, and integer literals such as a schedule
  // constant.  Pointers (ident, the slots) and the thread id are untouched —
  // they are not indices and their widths are their own.
  //
  // Where chunk_index is the induction variable's type, which is every runtime
  // but libgomp, all of this is the identity.
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

  // ---------------------------------------------------------------------------
  // Process plan.pre: emit each PlanCall directly; if `emit thread_bounds`
  // is encountered, materialise per-thread [lbThread, ubThread) inline via a
  // contiguous block-chunk formula.  The runtime path (e.g. iomp
  // __kmpc_for_static_init_4) has only PlanCalls in pre; the inline path
  // (PMSIS, libgomp) has only `emit thread_bounds`.  Whether `emit
  // thread_bounds` was seen drives the loop bounds choice below.
  // ---------------------------------------------------------------------------
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

    // Block work-distribution (one contiguous chunk per thread):
    //   trip  = ceil((ub - lb) / step)        -- iteration count
    //   chunk = ceil(trip / num_threads)      -- block size per thread
    //   lbThread = lb + threadId * chunk * step           (inclusive start)
    //   ubThread = min(lbThread + chunk * step, ub)         (exclusive end, clamped)
    // This avoids the per-thread DIVMOD (SDiv + SRem) of the balanced scheme:
    // a single ceiling-division computes the chunk and the upper bound is
    // clamped with a select.  The helper function names come from the DSL
    // properties thread_id_function / num_threads_function, so this code path
    // serves any runtime that exposes such helpers.
    std::string threadIdFn  = getStrProp("thread_id_function");
    std::string numThreadFn = getStrProp("num_threads_function");
    Value threadId   = emitNoArgI32Call(module, builder, loc, threadIdFn);
    Value numThreads = emitNoArgI32Call(module, builder, loc, numThreadFn);

    // trip = (ub - lb + step - 1) / step  [ceiling division of iteration count]
    Value range    = LLVM::SubOp::create(builder, loc, ub, lb);
    Value rangeS   = LLVM::AddOp::create(builder, loc, range,
                       LLVM::SubOp::create(builder, loc, step, one32));
    Value trip     = LLVM::SDivOp::create(builder, loc, rangeS, step);

    // chunk = ceil(trip / num_threads)
    Value tripPlusNC = LLVM::AddOp::create(builder, loc, trip,
                         LLVM::SubOp::create(builder, loc, numThreads, one32));
    Value chunk    = LLVM::SDivOp::create(builder, loc, tripPlusNC, numThreads);

    // lbThread = lb + threadId * chunk * step
    Value threadChunk  = LLVM::MulOp::create(builder, loc, threadId, chunk);
    Value threadOff    = LLVM::MulOp::create(builder, loc, threadChunk, step);
    lbThread           = LLVM::AddOp::create(builder, loc, lb, threadOff);

    // ubThread = lbThread + chunk * step, then clamp to ub
    Value chunkStep  = LLVM::MulOp::create(builder, loc, chunk, step);
    Value lbThreadEnd  = LLVM::AddOp::create(builder, loc, lbThread, chunkStep);
    Value clampCond  = LLVM::ICmpOp::create(builder, loc,
                         LLVM::ICmpPredicate::sgt, lbThreadEnd, ub);
    ubThread           = LLVM::SelectOp::create(builder, loc, clampCond, ub, lbThreadEnd);
    haveInlineBounds = true;
  }

  // Choose loop bounds and comparison predicate based on how pre populated them.
  // Inline block-chunking produces exclusive ubThread → use slt.
  // Runtime init (e.g. __kmpc_for_static_init_4) writes inclusive ub into pub
  // (we initialised pub with ub - step) → use sle.
  // A chunked construct has neither yet: its slots are filled once per chunk,
  // so the two are read in chunkBody below and the predicate comes from the
  // runtime's own convention for the bound it writes there.
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

  // Emit a chunk-acquisition block at the builder's current insertion point and
  // give back the i1 "there was work" test of its last call — the last one
  // because a `when`/`otherwise` chain collapses to exactly one call, and it is
  // that call's result the loop turns on.  The caller places the block: once as
  // the guard ahead of the loop, once as the latch that closes it.
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

  // A chunk block with no call in it leaves the loop with nothing to turn on.
  // Say so rather than emit a loop that never runs.
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

  // -------------------------------------------------------------------------
  // Build the loop.  A chunked construct wraps the sequential loop in an outer
  // one, rotated so the acquisition call sits in a guard and a latch:
  //
  //   pre:  first_chunk?  -> chunkBody : after
  //   chunkBody:          read the slots the call filled, then the inner loop
  //   loopHeader/Body/Latch                    (as in the unchunked case)
  //   chunkLatch:         next_chunk? -> chunkBody : after
  //
  // Rotating it is what lets the opening call differ from the repeat one, which
  // libgomp needs (start hands out the first chunk, next the rest) and iomp does
  // not — there the same call fills both places.
  // -------------------------------------------------------------------------
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

  builder.setInsertionPointToEnd(preBlock);
  Value pi = LLVM::AllocaOp::create(builder, loc, ptrT, iterTy, one64);
  if (isChunked) {
    // The opening chunk: first_chunk where the runtime's opening call differs
    // from the repeat one, next_chunk where it does not.
    Value more = emitChunkCall(
        plan.firstChunk.empty() ? plan.nextChunk : plan.firstChunk);
    LLVM::CondBrOp::create(builder, loc, more,
      chunkBody, mlir::ValueRange{}, afterBlock, mlir::ValueRange{});

    // Each chunk arrives in the slots; the loop runs over it in its own type.
    builder.setInsertionPointToEnd(chunkBody);
    loopStart = fromIdx(LLVM::LoadOp::create(builder, loc, chunkIdxTy, plb));
    loopEnd   = fromIdx(LLVM::LoadOp::create(builder, loc, chunkIdxTy, pub));
    LLVM::StoreOp::create(builder, loc, loopStart, pi);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);
  } else {
    LLVM::StoreOp::create(builder, loc, loopStart, pi);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loopHeader);
  }

  // Where the inner loop goes when the chunk is exhausted: back for another one,
  // or out.
  Block *loopExit = isChunked ? chunkLatch : afterBlock;

  builder.setInsertionPointToEnd(loopHeader);
  Value curI = LLVM::LoadOp::create(builder, loc, iterTy, pi);
  Value cond = LLVM::ICmpOp::create(builder, loc, cmpPred, curI, loopEnd);
  LLVM::CondBrOp::create(builder, loc, cond,
    loopBody, mlir::ValueRange{}, loopExit, mlir::ValueRange{});

  // Move the loop nest body into loopBody.
  // The nest region may have multiple blocks (from inner loops).
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
    // Multiple blocks (inner loops from CIR): splice ALL blocks before
    // loopLatch first, then move first block's ops into loopBody — this
    // preserves branch targets since blocks are already in the region.
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

  // The chunk is done: ask for another one, and leave when the answer is no.
  if (isChunked) {
    builder.setInsertionPointToEnd(chunkLatch);
    Value more = emitChunkCall(plan.nextChunk);
    LLVM::CondBrOp::create(builder, loc, more,
      chunkBody, mlir::ValueRange{}, afterBlock, mlir::ValueRange{});
  }

  // Emit post-loop calls (e.g. fini, optional barrier).
  builder.setInsertionPointToStart(afterBlock);
  emitPlanCalls(plan.post, builder);

  wsOp.erase();
  return success();
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

    // No barrier/taskwait plans are needed here any more: OmpToOmpLowerPass now
    // converts every barrier and taskwait — nested in a parallel or not — into a
    // ConstructOp carrying its own plan, and PlanLoweringPass emits the calls.
    // This pass only binds the thread id they cannot derive (bindGtidOnLeaves).

    // Step 1: outline parallel constructs.
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
      // The bounds by value, next to the in/out slots above: a dispatch API
      // takes them that way, and which upper bound it wants — the last valid
      // iteration or the one past it — is its own convention, so both are
      // offered and the rules pick.
      ctx["lower_val"]  = dsl::makeStr("%lb_val");
      ctx["upper_val"]  = dsl::makeStr("%ub_val");
      ctx["upper_incl"] = dsl::makeStr("%ub_incl");
      // The chunk size the clause asked for, absent when it named none — which
      // is what lets the rules write `when has(chunk)` and fall back to their
      // own default.
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

      // Mark the pass failed but carry on to the next loop: a rule file with a
      // mistake in it usually has the same mistake in every loop that reads it,
      // and reporting them one run at a time is a poor way to find that out.
      if (failed(lowerWsloop(wsOp, module, *plan)))
        signalPassFailure();
    }

    // The trailing team barrier of a parallel region, redundant with the join
    // of the fork call, is not dropped here: that rule lives in
    // OmpBarrierElimPass (--omp-barrier-elim), which states it on the omp
    // dialect, before a runtime is chosen, so one rule serves all three.
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::createOmpOutliningPass(std::string dslFile, std::string runtime) {
  return std::make_unique<OmpOutliningPass>(
      std::move(dslFile), std::move(runtime));
}
