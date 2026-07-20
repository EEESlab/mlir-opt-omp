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
#include "llvm/ADT/StringExtras.h"
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

static void replaceUsesInRegion(Region &region, Value oldVal, Value newVal) {
  for (auto &use : llvm::make_early_inc_range(oldVal.getUses()))
    if (region.isAncestor(use.getOwner()->getParentRegion()))
      use.set(newVal);
}

// The single clause operand carried by a ConstructOp, if any: num_threads for
// parallel, if_clause for task.  Which one it is follows from the construct
// kind and the symbolic arg name being resolved in the plan.
static Value getClauseOperand(ConstructOp op) {
  auto ops = op.getClauseOperands();
  return ops.empty() ? Value() : ops[0];
}

static std::string getPropStr(ConstructOp op, llvm::StringRef key) {
  auto dict = op.getPropDict();
  if (!dict) return "";
  if (auto sa = llvm::dyn_cast_or_null<StringAttr>(dict.get(key)))
    return sa.getValue().str();
  return "";
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

// KMP_IDENT_KMPC: the default ident flag, always set, mirroring
// OMPIRBuilder::getOrCreateIdent.  Idents with exactly these flags are the
// "default ident" that fork/gtid calls seed off and that call sites cache.
static constexpr uint32_t kIdentKmpc = 0x02;

// Map a DSL ident flag token to the effective `ident_t.flags` bitmask.
// Values match clang's OpenMPLocationFlags (CGOpenMPRuntime.cpp).
static uint32_t identFlagBits(llvm::StringRef tok) {
  uint32_t kmpc = kIdentKmpc;
  if (tok.empty() || tok == "kmpc")        return kmpc;
  if (tok == "barrier_expl")               return kmpc | 0x20;
  if (tok == "barrier_impl" ||
      tok == "barrier_impl_for")           return kmpc | 0x40;
  if (tok == "barrier_impl_sections")      return kmpc | 0xC0;
  if (tok == "barrier_impl_single")        return kmpc | 0x140;
  if (tok == "work_loop")                  return kmpc | 0x200;
  if (tok == "work_sections")              return kmpc | 0x400;
  if (tok == "work_distribute")            return kmpc | 0x800;
  return kmpc; // unknown token → plain KMPC
}

// Recognise a symbolic ident reference: "ident", "%ident", or "%ident:<flag>".
// On match, set `flagsOut` to the effective flag bits and return true.
static bool parseIdentRef(llvm::StringRef s, uint32_t &flagsOut) {
  if (!(s.consume_front("%ident") || s.consume_front("ident")))
    return false;
  llvm::StringRef flag = "";
  if (s.consume_front(":"))
    flag = s;
  else if (!s.empty())
    return false; // e.g. "identity" must not match
  flagsOut = identFlagBits(flag);
  return true;
}

// Get (creating on first use) the address of the `ident_t` global for a given
// flags value. Mirrors OMPIRBuilder::getOrCreateIdent: one private constant
// global per distinct flags value, all sharing a single default psource string
// ";unknown;unknown;0;0;;" (reserved_3 = its length, NUL excluded).
static Value getOrCreateIdent(ModuleOp module, OpBuilder &builder, Location loc,
                              MLIRContext *ctx, uint32_t flags) {
  auto i32t = IntegerType::get(ctx, 32);
  auto ptr  = ptrTy(ctx);

  // Shared default source-location string, NUL-terminated like
  // ConstantDataArray::getString. reserved_3 stores the length without NUL.
  llvm::StringRef srcName = "__omp_src_loc_default";
  const std::string srcText = ";unknown;unknown;0;0;;";
  if (!module.lookupSymbol(srcName)) {
    std::string data = srcText;
    data.push_back('\0');
    auto arrTy = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), data.size());
    OpBuilder gb(ctx);
    gb.setInsertionPointToStart(module.getBody());
    LLVM::GlobalOp::create(gb, loc, arrTy, /*isConstant=*/true,
      LLVM::Linkage::Private, srcName, StringAttr::get(ctx, data));
  }

  std::string identName = "__omp_ident_" + llvm::utohexstr(flags, true);
  if (!module.lookupSymbol(identName)) {
    auto identStructTy = LLVM::LLVMStructType::getLiteral(
      ctx, {i32t, i32t, i32t, i32t, ptr});
    OpBuilder gb(ctx);
    gb.setInsertionPointToStart(module.getBody());
    auto global = LLVM::GlobalOp::create(gb, loc, identStructTy,
      /*isConstant=*/true, LLVM::Linkage::Private, identName, Attribute{},
      /*alignment=*/8);
    global.setUnnamedAddr(LLVM::UnnamedAddr::Global);
    Block *initBlock = new Block();
    global.getInitializerRegion().push_back(initBlock);
    OpBuilder ib(ctx);
    ib.setInsertionPointToStart(initBlock);
    auto ci = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(ib, loc, i32t, IntegerAttr::get(i32t, v));
    };
    auto ins = [&](Value v, Value st, unsigned idx) -> Value {
      return LLVM::InsertValueOp::create(ib, loc, identStructTy, st, v,
        ArrayRef<int64_t>{(int64_t)idx});
    };
    Value s = LLVM::UndefOp::create(ib, loc, identStructTy);
    s = ins(ci(0),                       s, 0); // reserved_1
    s = ins(ci((int64_t)flags),          s, 1); // flags (incl. KMPC)
    s = ins(ci(0),                       s, 2); // reserved_2
    s = ins(ci((int64_t)srcText.size()), s, 3); // reserved_3 = strlen(psource)
    Value srcAddr = LLVM::AddressOfOp::create(ib, loc, ptr,
      FlatSymbolRefAttr::get(ctx, srcName));
    s = ins(srcAddr, s, 4);                     // psource
    LLVM::ReturnOp::create(ib, loc, s);
  }
  return LLVM::AddressOfOp::create(builder, loc, ptr,
    FlatSymbolRefAttr::get(ctx, identName));
}

// ---------------------------------------------------------------------------
// Symbolic token resolution
// ---------------------------------------------------------------------------
// Every lowering path resolves DSL argument tokens (strings like "%gtid",
// "ident", "%lb", "env_ptr", ...) to SSA Values against one shared vocabulary:
//   1. a site-specific binding (looked up first, so a construct's own tokens and
//      `let`-bound results shadow the built-ins);
//   2. an ident reference ("ident"/"%ident"/"%ident:<flag>") -> resolveIdent;
//   3. "%gtid"                                                -> resolveGtid;
//   4. anything unrecognised                                  -> an undef ptr.
// resolveIdent/resolveGtid are seams so each caller keeps its own caching and
// lazy-materialisation policy (e.g. the fork path only emits the gtid call when
// a token actually references it).  Literal (integer/bool) arguments stay in
// each caller's arg loop: most sites emit the same arith i32 constant, but
// wsloop types them as the induction variable (LLVM constant) and the packed
// invoke adds i8-bool / i64-size forms.
static Value resolveSymbolToken(
    llvm::StringRef s, OpBuilder &builder, Location loc,
    const llvm::StringMap<Value> &bindings,
    llvm::function_ref<Value(uint32_t)> resolveIdent,
    llvm::function_ref<Value()> resolveGtid) {
  if (auto it = bindings.find(s); it != bindings.end())
    return it->second;
  uint32_t flags;
  if (parseIdentRef(s, flags))
    return resolveIdent(flags);
  if (s == "%gtid")
    return resolveGtid();
  return LLVM::UndefOp::create(builder, loc, ptrTy(builder.getContext()));
}

// No-bindings overload for sites whose only symbolic tokens are ident/%gtid
// (e.g. barrier).
static Value resolveSymbolToken(
    llvm::StringRef s, OpBuilder &builder, Location loc,
    llvm::function_ref<Value(uint32_t)> resolveIdent,
    llvm::function_ref<Value()> resolveGtid) {
  static const llvm::StringMap<Value> noBindings;
  return resolveSymbolToken(s, builder, loc, noBindings, resolveIdent,
                            resolveGtid);
}

// Shared policy for the resolveIdent seam: default (KMPC) flags reuse the
// caller's cached default ident; any other flags get their own global
// (deduped by symbol name inside getOrCreateIdent).
static Value resolveIdentToken(uint32_t flags, ModuleOp module,
                               OpBuilder &builder, Location loc,
                               MLIRContext *ctx,
                               llvm::function_ref<Value()> defaultIdent) {
  return flags == kIdentKmpc
             ? defaultIdent()
             : getOrCreateIdent(module, builder, loc, ctx, flags);
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
// v1 limitations: no firstprivate/private clause wiring (captures only); no
// if/final clause (always deferred); task_flags = 1 (tied).
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
  // block args that arrived with the region; v1 does no firstprivate wiring, so
  // unused ones are dropped after capture unpacking.
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
    unpackCapturesFromBase(prologue, loc, one64, shareds, sharedsTy, captures,
      privateCaptures, scalarAllocaCaptures, ptrAllocaCaptures,
      outlinedFn.getBody());
  }

  // Drop stale privatizer block args (must be unused in v1).
  for (auto arg : llvm::reverse(staleArgs))
    if (arg.use_empty()) entry.eraseArgument(arg.getArgNumber());

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
    } else {
      auto decl = getOrInsertDeclWithReturn(module, callee, types, i32t, builder);
      func::CallOp::create(builder, loc, decl, args);
    }
  }

  op.erase();
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

  // capture_strategy is the single ABI discriminator: the delivery mechanism it
  // names uniquely entails the outlined-function signature.
  //   - by_pointer -> microtask   void(gtid, btid, cap0, cap1, ...)
  //   - packed     -> closure      void(ptr data)   (captures in one struct)
  //   - shareds    -> task routine i32(gtid, ptr task), captures via
  //                   task->shareds, emitted by outlineTaskEntry.
  enum class CaptureAbi { ByPointer, Packed, Shareds };
  std::string captureStrat = getPropStr(op, "capture_strategy");
  CaptureAbi abi;
  if (captureStrat == "by_pointer")   abi = CaptureAbi::ByPointer;
  else if (captureStrat == "packed")  abi = CaptureAbi::Packed;
  else if (captureStrat == "shareds") abi = CaptureAbi::Shareds;
  else {
    op.emitError("unknown capture_strategy '" + captureStrat +
                 "' (expected by_pointer, packed, or shareds)");
    return;
  }

  // iomp task uses a distinct ABI — the shareds signature: an
  // i32(i32 gtid, ptr task) entry whose captures live in a runtime-allocated
  // shareds block, emitted via the __kmpc_omp_task_alloc/task two-call
  // sequence.
  if (abi == CaptureAbi::Shareds) {
    outlineTaskEntry(op, module, counter);
    return;
  }

  bool isMicrotask = abi == CaptureAbi::ByPointer;
  bool isPacked    = abi == CaptureAbi::Packed;

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

  // Name the outlined function after the construct ("outlined_parallel_N",
  // "outlined_task_N", ...).  Parallel keeps its historical name.
  std::string fnName =
      "outlined_" + op.getConstructName().str() + "_" + std::to_string(counter++);

  // Build outlined function argument types.
  SmallVector<Type> fnArgTypes;
  if (isMicrotask) {
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
  eraseDeadCasts(outlinedFn);

  // Erase unused capture args from the entry block and filter captures list.
  // This removes captures that were only needed as privatizer sources
  // (e.g., source allocas for private vars that don't need copying).
  {
    unsigned capBase = isMicrotask ? 2 : 1; // microtask: [gtid,btid,...]; packed: [data,...]
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
  replaceTerminatorsWithReturn(outlinedFn, [](OpBuilder &tb, Location l) {
    func::ReturnOp::create(tb, l);
  });

  // Lower omp.barrier inside the outlined function — DSL-driven.
  // The barrier plan (one per runtime) is built once in runOnOperation;
  // here we only resolve the symbolic args (%ident, %gtid) at each call site.
  SmallVector<Operation *> barriers;
  for (auto &block : outlinedFn.getBody())
    for (auto &innerOp : block)
      if (innerOp.getName().getStringRef() == "omp.barrier")
        barriers.push_back(&innerOp);
  for (auto *barrierOp : barriers) {
    OpBuilder bb(barrierOp);
    Location bloc = barrierOp->getLoc();

    auto resolveArg = [&](const dsl::Value &v) -> Value {
      if (auto *sv = std::get_if<dsl::StrVal>(&v))
        return resolveSymbolToken(
            sv->value, bb, bloc,
            [&](uint32_t flags) {
              return getOrCreateIdent(module, bb, bloc, ctx, flags);
            },
            [&]() -> Value {
              // Microtask convention: first arg is ptr to i32 gtid.
              auto &fnEntry = outlinedFn.getBody().front();
              if (fnEntry.getNumArguments() >= 2)
                return LLVM::LoadOp::create(bb, bloc, i32Ty(ctx),
                                            fnEntry.getArgument(0));
              return LLVM::UndefOp::create(bb, bloc, i32Ty(ctx));
            });
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
    if (Value ct = getClauseOperand(op)) preBindings["num_threads"] = ct;

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

    if (isPacked) {
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
      llvm::StringMap<Value> invokeBindings;
      invokeBindings["body"]              = fnPtrCast;
      invokeBindings["outlined_parallel"] = fnPtrCast;
      invokeBindings["outlined_task"]     = fnPtrCast;
      invokeBindings["env_ptr"]           = structAlloca;
      if (Value ct = getClauseOperand(op)) invokeBindings["num_threads"] = ct;

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
            else if (s == "if_clause" && getClauseOperand(op)) {
              // Normalise the if-clause SSA value (typically i1) to i8 to
              // match the C `_Bool` parameter of GOMP_task.
              Value ifv = getClauseOperand(op);
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

    // Pre-compute the barrier plan once; the same plan is reused for every
    // omp.barrier inside outlined parallel functions.
    llvm::StringMap<dsl::Value> barrierCtx;
    barrierCtx["ident"]      = dsl::makeStr("%ident");
    barrierCtx["global_tid"] = dsl::makeStr("%gtid");
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
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::createOmpOutliningPass(std::string dslFile, std::string runtime) {
  return std::make_unique<OmpOutliningPass>(
      std::move(dslFile), std::move(runtime));
}
