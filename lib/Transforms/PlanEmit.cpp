// PlanEmit.cpp
//
// Definitions for the shared plan-emission vocabulary.  Moved here verbatim
// from OmpOutliningPass.cpp, which was their only home while it was also the
// only pass that could resolve them.

#include "PlanEmit.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::omp_lower;

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

Type mlir::omp_lower::ptrTy(MLIRContext *ctx) {
  return LLVM::LLVMPointerType::get(ctx);
}

Type mlir::omp_lower::i32Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 32);
}

// ---------------------------------------------------------------------------
// Runtime declarations
// ---------------------------------------------------------------------------

func::FuncOp mlir::omp_lower::getOrInsertDecl(ModuleOp module,
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

func::FuncOp mlir::omp_lower::getOrInsertDeclWithReturn(ModuleOp module,
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

// ---------------------------------------------------------------------------
// ident_t
// ---------------------------------------------------------------------------

uint32_t mlir::omp_lower::identFlagBits(llvm::StringRef tok) {
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

bool mlir::omp_lower::parseIdentRef(llvm::StringRef s, uint32_t &flagsOut) {
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

Value mlir::omp_lower::getOrCreateIdent(ModuleOp module, OpBuilder &builder,
                                        Location loc, MLIRContext *ctx,
                                        uint32_t flags) {
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

Value mlir::omp_lower::resolveIdentToken(uint32_t flags, ModuleOp module,
                                         OpBuilder &builder, Location loc,
                                         MLIRContext *ctx,
                                         llvm::function_ref<Value()> defaultIdent) {
  return flags == kIdentKmpc
             ? defaultIdent()
             : getOrCreateIdent(module, builder, loc, ctx, flags);
}

// ---------------------------------------------------------------------------
// Symbolic token resolution
// ---------------------------------------------------------------------------

Value mlir::omp_lower::resolveSymbolToken(
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

Value mlir::omp_lower::resolveSymbolToken(
    llvm::StringRef s, OpBuilder &builder, Location loc,
    llvm::function_ref<Value(uint32_t)> resolveIdent,
    llvm::function_ref<Value()> resolveGtid) {
  static const llvm::StringMap<Value> noBindings;
  return resolveSymbolToken(s, builder, loc, noBindings, resolveIdent,
                            resolveGtid);
}
