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

// Map a DSL ABI type name (as produced by the `struct(...)` token) to an MLIR
// type.  Kept small on purpose: extend as new layouts need more field types.
static Type parseAbiType(MLIRContext *ctx, llvm::StringRef t) {
  if (t == "ptr") return ptrTy(ctx);
  if (t == "i32") return i32Ty(ctx);
  if (t == "i64") return IntegerType::get(ctx, 64);
  if (t == "i8")  return IntegerType::get(ctx, 8);
  return ptrTy(ctx);
}

// Read a DSL-owned struct-layout property of the form "%struct:t0,t1,..." into
// an LLVM literal struct type.  Absent or malformed property falls back to the
// caller's default so older DSL files keep working.
static LLVM::LLVMStructType getPropStructType(ConstructOp op, llvm::StringRef key,
    MLIRContext *ctx, LLVM::LLVMStructType fallback) {
  std::string s = getPropStr(op, key);
  llvm::StringRef body(s);
  if (!body.consume_front("%struct:")) return fallback;
  SmallVector<llvm::StringRef> toks;
  body.split(toks, ',');
  SmallVector<Type> fields;
  for (auto tok : toks) {
    tok = tok.trim();
    if (!tok.empty()) fields.push_back(parseAbiType(ctx, tok));
  }
  if (fields.empty()) return fallback;
  return LLVM::LLVMStructType::getLiteral(ctx, fields);
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

// At the call site (builder `b`), store each capture into the struct at
// `structBase`.  Mirrors unpackCapturesFromBase: a private slot is left undef,
// a scalar/ptr alloca is loaded by value, a plain capture is stored as-is.
static void storeCapturesToBase(
    OpBuilder &b, Location loc, Value structBase,
    LLVM::LLVMStructType structTy, ArrayRef<Value> captures,
    ArrayRef<Type> fieldTypes,
    const llvm::SetVector<Value> &privateCaptures,
    const llvm::SetVector<Value> &scalarAllocaCaptures,
    const llvm::SetVector<Value> &ptrAllocaCaptures) {
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
    Value gep = LLVM::GEPOp::create(b, loc, ptr, structTy, structBase,
      ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
    LLVM::StoreOp::create(b, loc, capVal, gep);
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
// Lowers an omp_lower.construct for an iomp task:
//   - outlines the body into an entry function  i32(i32 gtid, ptr task) that
//     loads task->shareds and unpacks the captures from it;
//   - at the call site emits __kmpc_omp_task_alloc, populates task->shareds
//     with the captures, then __kmpc_omp_task.
// An omp.taskwait nested in the task body is lowered here too (via the shared
// leaf helper) when the runtime provides a taskwait plan; %gtid is the entry's
// arg 0 (an i32 value, unlike the microtask ptr-to-i32 convention).
// Supports implicit captures and explicit firstprivate clauses (copy-in into a
// task-private slot, mirroring the packed path), and the if clause via a
// runtime branch: deferred __kmpc_omp_task when the condition holds, the
// undeferred __kmpc_omp_task_begin_if0 / direct entry call / _complete_if0
// protocol when it does not.  v1 limitations: pure `private` clauses are not
// wired (diagnosed, not miscompiled); no final clause; task_flags = 1 (tied);
// depend/nowait on taskwait ignored; omp.barrier is illegal in a task region
// and diagnosed.
static void bindGtidOnLeaves(func::FuncOp outlinedFn, MLIRContext *ctx,
                             llvm::function_ref<Value(OpBuilder &, Location)>
                                 makeGtid);

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
  builder.setInsertionPoint(op);

  // ident and global_tid are resolved on demand from the invoke args (same
  // model as parallel/barrier/wsloop — no DSL `emit`).  Unlike parallel, where
  // gtid is optional, every task invoke call uses both, so materialising them
  // eagerly here is equivalent to lazy and keeps the code simpler: one ident
  // global and one global_tid_function call (DSL property, iomp:
  // __kmpc_global_thread_num) shared by the two invoke calls.  gtid seeds off
  // the default (KMPC-flagged) ident, matching the runtime's contract.
  Value identVal = getOrCreateIdent(module, builder, loc, ctx, kIdentKmpc);
  std::string gtidFnName = getPropStr(op, "global_tid_function");
  Value gtid;
  if (gtidFnName.empty()) {
    // No global_tid_function DSL property (non-iomp runtime): there is no gtid
    // source, and declaring a nameless function would fail the verifier.
    op.emitWarning("task lowering: runtime defines no global_tid_function; "
                   "using undef gtid");
    gtid = LLVM::UndefOp::create(builder, loc, i32t);
  } else {
    auto gtidDecl = getOrInsertDeclWithReturn(module,
      gtidFnName, {ptr}, i32t, builder);
    gtid = func::CallOp::create(builder, loc, gtidDecl,
      ValueRange{identVal}).getResult(0);
  }

  DataLayout dl(module);
  uint64_t taskSize    = dl.getTypeSize(kmpTaskTy).getFixedValue();
  uint64_t sharedsSize = dl.getTypeSize(sharedsTy).getFixedValue();
  Value taskSizeV = LLVM::ConstantOp::create(builder, loc, i64t,
    IntegerAttr::get(i64t, (int64_t)taskSize));
  Value sharedsSizeV = LLVM::ConstantOp::create(builder, loc, i64t,
    IntegerAttr::get(i64t, (int64_t)sharedsSize));

  Value fnPtr = func::ConstantOp::create(builder, loc,
    outlinedFn.getFunctionType(), FlatSymbolRefAttr::get(ctx, fnName));
  Value fnPtrCast = UnrealizedConversionCastOp::create(builder, loc,
    TypeRange{ptr}, ValueRange{fnPtr}).getResult(0);

  // Emit the invoke actions in DSL order.  A `let <name> = call ...` binds the
  // call's SSA result under the token "%<name>"; later arg tokens and the
  // `populate_shareds` verb resolve against this map (Approach B, see
  // docs/dsl-result-binding-proposal.md).  The construct's built-in tokens
  // (task_size/shareds_size/body) seed the same map; because the map lookup
  // runs first, a `let` result shadows a built-in of the same name.
  // task_flags is not a symbolic token: it is a `let task_flags = 1;` in
  // rules.dsl, so it reaches the invoke as an IntegerAttr and is handled by the
  // integer branch of the arg loop below.
  llvm::StringMap<Value> boundResults;
  boundResults["task_size"]    = taskSizeV;
  boundResults["shareds_size"] = sharedsSizeV;
  boundResults["body"]         = fnPtrCast;

  // Resolve one symbolic string arg to an SSA value (see resolveSymbolToken).
  // The default (KMPC) ident reuses the value materialised above; %gtid is the
  // gtid emitted eagerly at the call site (every task invoke call needs it).
  auto resolveToken = [&](llvm::StringRef s) -> Value {
    return resolveSymbolToken(
        s, builder, loc, boundResults,
        [&](uint32_t flags) {
          return resolveIdentToken(flags, module, builder, loc, ctx,
                                   [&] { return identVal; });
        },
        [&] { return gtid; });
  };

  for (auto attr : op.getInvoke()) {
    // `emit populate_shareds(<task>)`: write the captures into task->shareds.
    if (auto ea = llvm::dyn_cast<PlanEmitAttr>(attr)) {
      if (ea.getSymName().getValue() != "populate_shareds") continue;
      if (captures.empty()) continue;
      Value base;
      if (auto sv = llvm::dyn_cast<StringAttr>(ea.getValue()))
        base = resolveToken(sv.getValue());
      else
        base = LLVM::UndefOp::create(builder, loc, ptr);
      Value sg = LLVM::GEPOp::create(builder, loc, ptr, kmpTaskTy, base,
        ArrayRef<LLVM::GEPArg>{0, 0});
      Value sh = LLVM::LoadOp::create(builder, loc, ptr, sg);
      storeCapturesToBase(builder, loc, sh, sharedsTy, captures, fieldTypes,
        privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures);
      continue;
    }

    auto ca = llvm::dyn_cast<PlanCallAttr>(attr);
    if (!ca) continue;
    llvm::StringRef callee = ca.getCallee().getValue();
    SmallVector<Value> args;
    SmallVector<Type>  types;
    for (auto argAttr : ca.getArgs()) {
      Value v;
      if (auto sa = llvm::dyn_cast<StringAttr>(argAttr)) {
        v = resolveToken(sa.getValue());
      } else if (auto ia = llvm::dyn_cast<IntegerAttr>(argAttr)) {
        v = arith::ConstantOp::create(builder, loc, i32t,
          IntegerAttr::get(i32t, ia.getInt()));
      } else {
        v = LLVM::UndefOp::create(builder, loc, ptr);
      }
      args.push_back(v);
      types.push_back(v.getType());
    }

    // A bound call returns a handle (ptr); a fire-and-forget call returns i32
    // and its result is dropped.
    if (auto resultName = ca.getResult()) {
      auto decl = getOrInsertDeclWithReturn(module, callee, types, ptr, builder);
      Value res = func::CallOp::create(builder, loc, decl, args).getResult(0);
      boundResults["%" + resultName.getValue().str()] = res;
    } else if (callee == "__kmpc_omp_task" && !args.empty() &&
               getClauseOperand(op, "if_clause")) {
      // if(cond) needs a branch on the runtime value (the flat DSL plan can
      // only branch on clause *presence*), so it is emitted here:
      //   cond true  → deferred:   __kmpc_omp_task(ident, gtid, task)
      //   cond false → undeferred: the if0 protocol — begin_if0, direct call
      //                of the entry on the current thread, complete_if0.
      // The task is allocated either way (the if0 calls also take kmp_task_t*).
      // The task handle is the call's last resolved arg (the DSL `task`
      // binding in `call "__kmpc_omp_task"(ident, global_tid, task)`).
      Value taskV = args.back();
      Value cond = clauseToI1(builder, loc, getClauseOperand(op, "if_clause"));
      Block *curBlock = builder.getInsertionBlock();
      // op and everything after it continue in contBlock.
      Block *contBlock = curBlock->splitBlock(op);
      Block *deferredBlock = builder.createBlock(contBlock);
      Block *if0Block = builder.createBlock(contBlock);

      builder.setInsertionPointToEnd(curBlock);
      LLVM::CondBrOp::create(builder, loc, cond, deferredBlock, if0Block);

      builder.setInsertionPointToEnd(deferredBlock);
      auto decl = getOrInsertDeclWithReturn(module, callee, types, i32t, builder);
      func::CallOp::create(builder, loc, decl, args);
      LLVM::BrOp::create(builder, loc, contBlock);

      builder.setInsertionPointToEnd(if0Block);
      auto beginDecl = getOrInsertDecl(module, "__kmpc_omp_task_begin_if0",
        {ptr, i32t, ptr}, builder);
      func::CallOp::create(builder, loc, beginDecl,
        ValueRange{identVal, gtid, taskV});
      func::CallOp::create(builder, loc, outlinedFn,
        ValueRange{gtid, taskV});
      auto completeDecl = getOrInsertDecl(module, "__kmpc_omp_task_complete_if0",
        {ptr, i32t, ptr}, builder);
      func::CallOp::create(builder, loc, completeDecl,
        ValueRange{identVal, gtid, taskV});
      LLVM::BrOp::create(builder, loc, contBlock);

      builder.setInsertionPoint(op);
    } else {
      auto decl = getOrInsertDeclWithReturn(module, callee, types, i32t, builder);
      func::CallOp::create(builder, loc, decl, args);
    }
  }

  op.erase();
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

  // ---- Emit the runtime call at the call site ----
  std::string runtimeCallee;
  for (auto attr : op.getInvoke())
    if (auto ca = llvm::dyn_cast<PlanCallAttr>(attr)) {
      runtimeCallee = ca.getCallee().getValue().str();
      break;
    }

  // Set by whichever emission path consumes the if clause; checked after
  // emission so an unconsumed if is flagged instead of silently dropped.
  bool ifClauseUsed = false;

  if (!runtimeCallee.empty()) {
    builder.setInsertionPoint(op);

    SmallVector<Value> callArgs;
    SmallVector<Type>  callTypes;

    // Call-site ident and master global_tid are materialised on demand — once,
    // the first time a pre/invoke call actually references them — with no `emit`
    // declaration in the DSL (same on-demand model as barrier/wsloop/task).
    //   - ident: needed by the iomp fork and as the gtid seed; the packed path
    //     (libgomp/pmsis) references neither, so it is never emitted there.
    //   - global_tid: needed only when an optional push_num_threads /
    //     push_proc_bind pre-call is present.  Its function name comes from the
    //     `global_tid_function` DSL property; it seeds off the default ident.
    Value identCache;
    auto getIdent = [&]() -> Value {
      if (!identCache)
        identCache = getOrCreateIdent(module, builder, loc, ctx, kIdentKmpc);
      return identCache;
    };
    Value gtidCache;
    auto getGtid = [&]() -> Value {
      if (!gtidCache) {
        std::string gtidFnName = getPropStr(op, "global_tid_function");
        if (gtidFnName.empty()) {
          // A %gtid token under a runtime with no global_tid_function DSL
          // property (libgomp/pmsis) has no gtid source; declaring a nameless
          // function would fail the verifier.
          op.emitWarning("'%gtid' referenced but the runtime defines no "
                         "global_tid_function; using undef");
          gtidCache = LLVM::UndefOp::create(builder, loc, i32Ty(ctx));
        } else {
          auto gtidDecl = getOrInsertDeclWithReturn(module,
            gtidFnName, {ptrTy(ctx)}, i32Ty(ctx), builder);
          gtidCache = func::CallOp::create(builder, loc, gtidDecl,
            ValueRange{getIdent()}).getResult(0);
        }
      }
      return gtidCache;
    };
    // Shared ident seam for the pre/invoke resolvers below (see
    // resolveIdentToken): the default ident reuses the cached fork ident.
    auto resolveIdent = [&](uint32_t flags) -> Value {
      return resolveIdentToken(flags, module, builder, loc, ctx, getIdent);
    };

    // Bindings for the pre-block resolver: num_threads when the clause is set.
    llvm::StringMap<Value> preBindings;
    if (Value ct = getClauseOperand(op, "num_threads"))
      preBindings["num_threads"] = ct;

    // Emit pre-block calls (push_num_threads, push_proc_bind, etc.)
    for (auto attr : op.getPre()) {
      auto ca = llvm::dyn_cast<PlanCallAttr>(attr);
      if (!ca) continue;
      SmallVector<Value> preArgs;
      SmallVector<Type>  preTypes;
      for (auto argAttr : ca.getArgs()) {
        Value v;
        if (auto sa = llvm::dyn_cast<StringAttr>(argAttr)) {
          v = resolveSymbolToken(sa.getValue(), builder, loc, preBindings,
                                 resolveIdent, getGtid);
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

    if (abi == CaptureAbi::Packed) {
      // -----------------------------------------------------------------------
      // PACKED / CLOSURE: build capture struct on the stack, pass ptr to it
      // -----------------------------------------------------------------------
      // Scalar/ptr-alloca captures are packed by value (see buildCaptureStruct
      // / storeCapturesToBase); the struct type is also needed below for
      // env_size / env_align (task).
      SmallVector<Type> fieldTypes;
      auto structTy =
          buildCaptureStruct(ctx, captures, scalarAllocaCaptures, fieldTypes);

      // Build the capture struct (same for all packed runtimes).
      Value structAlloca;
      if (!captures.empty()) {
        Value one64 = LLVM::ConstantOp::create(builder, loc,
          IntegerType::get(ctx, 64),
          IntegerAttr::get(IntegerType::get(ctx, 64), 1));
        structAlloca = LLVM::AllocaOp::create(builder, loc,
          ptrTy(ctx), structTy, one64);
        storeCapturesToBase(builder, loc, structAlloca, structTy, captures,
          fieldTypes, privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures);
      } else {
        structAlloca = LLVM::UndefOp::create(builder, loc, ptrTy(ctx));
      }

      // outlined fn pointer
      Value fnPtr = func::ConstantOp::create(builder, loc,
        outlinedFn.getFunctionType(),
        FlatSymbolRefAttr::get(ctx, fnName));
      Value fnPtrCast = UnrealizedConversionCastOp::create(
        builder, loc, TypeRange{ptrTy(ctx)}, ValueRange{fnPtr}).getResult(0);

      // Capture-struct size/alignment for GOMP_task's `long arg_size,
      // long arg_align` (libgomp memcpys arg_size bytes into task-private
      // memory when cpyfn is NULL).  Computed via the module DataLayout;
      // alignment falls back to 16 (a valid power-of-2 ≥ any field's align).
      auto i64t = IntegerType::get(ctx, 64);
      auto i8t  = IntegerType::get(ctx, 8);
      DataLayout dataLayout(module);
      uint64_t envSize  = captures.empty()
        ? 0u : dataLayout.getTypeSize(structTy).getFixedValue();
      uint64_t envAlign = captures.empty()
        ? 1u : dataLayout.getTypeABIAlignment(structTy);
      if (envAlign == 0) envAlign = 16;

      // Plain-lookup tokens shared with resolveSymbolToken.  env_size/env_align
      // (lazy i64 constants, emitted only when referenced so the parallel path
      // materialises none), if_clause (i1→i8 normalisation) and null (a real
      // null ptr, not the undef fallback) stay special-cased below.
      // `lower_in = plan` hands the invoke over to PlanLoweringPass instead of
      // emitting it here: this pass attaches the artifacts only it can produce
      // — the outlined function pointer and the capture struct — as named
      // operands, and leaves the construct standing for the plan pass to
      // consume.  That is the end state for every construct; the ones without
      // the property still emit below, because their invoke needs something the
      // flat plan cannot yet say (a variadic capture splice, a call whose
      // callee is a bound value, a branch on a clause).
      if (getPropStr(op, "lower_in") == "plan") {
        SmallVector<Value> operands(op.getClauseOperands().begin(),
                                    op.getClauseOperands().end());
        SmallVector<Attribute> names;
        if (auto existing = op.getClauseNames())
          names.assign(existing->begin(), existing->end());
        auto bind = [&](llvm::StringRef n, Value v) {
          operands.push_back(v);
          names.push_back(StringAttr::get(ctx, n));
        };
        // Bind every spelling the plan may use for the outlined function: the
        // context seeds `body` with the string "outlined_parallel"/"outlined_task",
        // so that — not "body" — is what reaches the plan as the argument.
        // Missing one resolves to an undef pointer and forks into nothing.
        bind("body", fnPtrCast);
        bind("outlined_parallel", fnPtrCast);
        bind("outlined_task", fnPtrCast);
        bind("env_ptr", structAlloca);
        op->setOperands(operands);
        op.setClauseNamesAttr(ArrayAttr::get(ctx, names));
        // Same guard as the emitting paths below, but the plan may well
        // consume the clause itself now (libgomp branches on it), so only warn
        // when nothing in the plan names it.
        if (getClauseOperand(op, "if_clause") &&
            !planNamesToken(op, "if_clause"))
          op.emitWarning("omp-outline: `if` clause is not supported by this "
                         "runtime/construct lowering and was ignored");
        return;   // deliberately not erased: the plan pass consumes it
      }

      llvm::StringMap<Value> invokeBindings;
      invokeBindings["body"]              = fnPtrCast;
      invokeBindings["outlined_parallel"] = fnPtrCast;
      invokeBindings["outlined_task"]     = fnPtrCast;
      invokeBindings["env_ptr"]           = structAlloca;
      if (Value ct = getClauseOperand(op, "num_threads"))
        invokeBindings["num_threads"] = ct;

      // Build call args from DSL invoke args, resolving symbolic names.
      for (auto attr : op.getInvoke()) {
        auto ca = llvm::dyn_cast<PlanCallAttr>(attr);
        if (!ca) continue;
        for (auto argAttr : ca.getArgs()) {
          Value v;
          if (auto sa = llvm::dyn_cast<StringAttr>(argAttr)) {
            llvm::StringRef s = sa.getValue();
            if (s == "env_size")
              v = LLVM::ConstantOp::create(builder, loc, i64t,
                    IntegerAttr::get(i64t, (int64_t)envSize));
            else if (s == "env_align")
              v = LLVM::ConstantOp::create(builder, loc, i64t,
                    IntegerAttr::get(i64t, (int64_t)envAlign));
            else if (s == "if_clause" && getClauseOperand(op, "if_clause")) {
              // Normalise the if-clause SSA value (typically i1) to i8 to
              // match the C `_Bool` parameter of GOMP_task.
              ifClauseUsed = true;
              Value ifv = getClauseOperand(op, "if_clause");
              if (ifv.getType() != i8t) {
                unsigned bw = ifv.getType().getIntOrFloatBitWidth();
                ifv = bw < 8
                  ? LLVM::ZExtOp::create(builder, loc, i8t, ifv).getResult()
                  : LLVM::TruncOp::create(builder, loc, i8t, ifv).getResult();
              }
              v = ifv;
            } else if (s == "null")
              v = LLVM::ZeroOp::create(builder, loc, ptrTy(ctx));
            else
              v = resolveSymbolToken(s, builder, loc, invokeBindings,
                                     resolveIdent, getGtid);
          } else if (auto ba = llvm::dyn_cast<BoolAttr>(argAttr)) {
            v = LLVM::ConstantOp::create(builder, loc, i8t,
                  IntegerAttr::get(i8t, ba.getValue() ? 1 : 0));
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
      callArgs.push_back(getIdent()); callTypes.push_back(ptrTy(ctx));

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

    // (The libgomp `parallel if(cond)` handling used to sit here, forcing
    // num_threads to 1 on the false side with an arith.select.  It is now a
    // `branch` in rules.dsl and emitted by PlanLoweringPass — the first case
    // where the DSL expresses a choice on a runtime value instead of C++.)

    // __kmpc_fork_call is variadic: (ident, argc, fn, ...captures) -> void.
    // Declare it as llvm.func variadic so multiple parallel regions with
    // different capture counts can share the same declaration.
    if (runtimeCallee == "__kmpc_fork_call") {
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

      Value ifCond = abi == CaptureAbi::Packed
                         ? Value()
                         : getClauseOperand(op, "if_clause");
      if (!ifCond) {
        LLVM::CallOp::create(builder, loc, llvmDecl, callArgs);
      } else {
        // parallel if(cond): branch on the runtime value, clang-style.
        //   cond true  → __kmpc_fork_call as usual
        //   cond false → serialized parallel: __kmpc_serialized_parallel,
        //                direct microtask call on this thread (gtid/btid
        //                passed by pointer, btid = 0),
        //                __kmpc_end_serialized_parallel.
        // __kmpc_fork_call_if is deliberately NOT used: it takes a single
        // packed `void *args` (argc <= 1), incompatible with the by_pointer
        // capture convention that passes each capture as its own vararg.
        ifClauseUsed = true;
        Value gtidVal = getGtid();
        auto i64t = IntegerType::get(ctx, 64);
        Value one64 = LLVM::ConstantOp::create(builder, loc, i64t,
          IntegerAttr::get(i64t, 1));
        Value zero32 = LLVM::ConstantOp::create(builder, loc, i32Ty(ctx),
          IntegerAttr::get(i32Ty(ctx), 0));
        Value gtidAddr = LLVM::AllocaOp::create(builder, loc, ptrTy(ctx),
          i32Ty(ctx), one64);
        Value btidAddr = LLVM::AllocaOp::create(builder, loc, ptrTy(ctx),
          i32Ty(ctx), one64);
        LLVM::StoreOp::create(builder, loc, gtidVal, gtidAddr);
        LLVM::StoreOp::create(builder, loc, zero32, btidAddr);
        Value cond = clauseToI1(builder, loc, ifCond);

        Block *curBlock = builder.getInsertionBlock();
        // op and everything after it continue in contBlock.
        Block *contBlock = curBlock->splitBlock(op);
        Block *forkBlock = builder.createBlock(contBlock);
        Block *serBlock  = builder.createBlock(contBlock);

        builder.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(builder, loc, cond, forkBlock, serBlock);

        builder.setInsertionPointToEnd(forkBlock);
        LLVM::CallOp::create(builder, loc, llvmDecl, callArgs);
        LLVM::BrOp::create(builder, loc, contBlock);

        builder.setInsertionPointToEnd(serBlock);
        auto serDecl = getOrInsertDecl(module, "__kmpc_serialized_parallel",
          {ptrTy(ctx), i32Ty(ctx)}, builder);
        func::CallOp::create(builder, loc, serDecl,
          ValueRange{getIdent(), gtidVal});
        // callArgs[3..] are exactly the capture args of the microtask.
        SmallVector<Value> directArgs{gtidAddr, btidAddr};
        for (size_t ai = 3; ai < callArgs.size(); ++ai)
          directArgs.push_back(callArgs[ai]);
        func::CallOp::create(builder, loc, outlinedFn, directArgs);
        auto endSerDecl = getOrInsertDecl(module,
          "__kmpc_end_serialized_parallel", {ptrTy(ctx), i32Ty(ctx)}, builder);
        func::CallOp::create(builder, loc, endSerDecl,
          ValueRange{getIdent(), gtidVal});
        LLVM::BrOp::create(builder, loc, contBlock);

        builder.setInsertionPoint(op);
      }
    } else {
      auto decl = getOrInsertDecl(module, runtimeCallee, callTypes, builder);
      func::CallOp::create(builder, loc, decl, callArgs);
    }
  }

  // An if clause that no emission path consumed would silently change
  // semantics (if(false) must run the region undeferred/serialized): flag it.
  if (getClauseOperand(op, "if_clause") && !ifClauseUsed)
    op.emitWarning("omp-outline: `if` clause is not supported by this "
                   "runtime/construct lowering and was ignored");

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

  // Get gtid from enclosing outlined function. (idents are resolved per call
  // argument below, since init/fini and the trailing barrier use different
  // flags.)
  // For iomp microtask: arg0 = ptr gtid, arg1 = ptr btid → load i32 from arg0.
  // For libgomp closure: arg0 = ptr data (capture struct) — no gtid ptr.
  //   Use omp_get_thread_num() to obtain the current thread id.
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

  // Map symbolic DSL names to SSA values.  The loop-specific slots (%lb, %ub,
  // %step, %stride, %last) are the site bindings; ident and %gtid come from the
  // shared vocabulary (see resolveSymbolToken).
  llvm::StringMap<Value> wsBindings;
  wsBindings["%lb"]     = plb;      // in/out lower-bound slot
  wsBindings["%ub"]     = pub;      // in/out upper-bound slot
  wsBindings["%step"]   = step;     // actual loop step
  wsBindings["%stride"] = pstride;  // output ptr for runtime stride
  wsBindings["%last"]   = plast;    // in/out last-iteration flag
  auto resolveCallArg = [&](const dsl::Value &v) -> Value {
    if (auto *sv = std::get_if<dsl::StrVal>(&v))
      return resolveSymbolToken(
          sv->value, builder, loc, wsBindings,
          [&](uint32_t flags) {
            return getOrCreateIdent(module, builder, loc, ctx, flags);
          },
          [&] { return gtidVal; });
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
  Value loopStart, loopEnd;
  LLVM::ICmpPredicate cmpPred;
  if (haveInlineBounds) {
    loopStart = lbThread;
    loopEnd   = ubThread;
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
      // Removed: no construct references a bare `chunk`; the static schedule's
      // chunk size flows from the runtime-level `let default_chunk` in rules.dsl.
      // ctx["chunk"]      = dsl::makeInt(1);
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

    // Step 3: drop the redundant trailing team barrier of each outlined
    // parallel region.  A team barrier that is the last operation before the
    // microtask returns is already covered by the implicit join barrier of
    // __kmpc_fork_call (the master waits for the whole team before fork_call
    // returns).  This is the combined `parallel for` case — clang elides the
    // work-sharing barrier the same way.  Removing it saves one global barrier
    // per parallel region, which is significant for kernels that fork many
    // short regions (e.g. stencils looping over thousands of time steps).
    //
    // A trailing barrier comes in two shapes at this point.  An explicit
    // omp.barrier is still an unlowered ConstructOp — PlanLoweringPass turns it
    // into calls later — and is recognised by its construct name.  A wsloop's
    // implicit barrier was emitted here by lowerWsloop as a real call, so it is
    // recognised by callee; that name is runtime-specific (iomp __kmpc_barrier,
    // libgomp GOMP_barrier, ...) and comes from the barrier plan.  A runtime
    // without a `construct barrier` simply has nothing to match.
    //
    // Only the *last* barrier before the return is dropped, so a barrier that
    // separates two work-sharing loops inside the same region is preserved.
    // Tasks are not fork/joined this way, so only outlined_parallel_* qualify.
    std::string barrierCallee;
    {
      llvm::StringMap<dsl::Value> barrierCtx;
      barrierCtx["ident"]      = dsl::makeStr("%ident");
      barrierCtx["global_tid"] = dsl::makeStr("%gtid");
      auto p = evaluator.buildPlan(runtimeName, "barrier", barrierCtx);
      if (p) {
        for (auto &action : p->invoke)
          if (auto *ca = std::get_if<dsl::PlanCall>(&action)) {
            barrierCallee = ca->callee;
            break;
          }
      } else {
        llvm::consumeError(p.takeError());
      }
    }
    module.walk([&](func::FuncOp fn) {
      if (!fn.getName().starts_with("outlined_parallel_"))
        return;
      for (Block &blk : fn.getBody()) {
        auto ret = llvm::dyn_cast_or_null<func::ReturnOp>(blk.getTerminator());
        if (!ret)
          continue;
        Operation *prev = ret->getPrevNode();
        if (!prev)
          continue;
        if (auto c = llvm::dyn_cast<ConstructOp>(prev)) {
          if (c.getConstructName() == "barrier")
            c.erase();
          continue;
        }
        auto call = llvm::dyn_cast<func::CallOp>(prev);
        if (call && !barrierCallee.empty() && call.getCallee() == barrierCallee)
          call.erase();
      }
    });
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::createOmpOutliningPass(std::string dslFile, std::string runtime) {
  return std::make_unique<OmpOutliningPass>(
      std::move(dslFile), std::move(runtime));
}
