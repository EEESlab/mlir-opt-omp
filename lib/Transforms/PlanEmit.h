// Vocabulary shared by OmpOutliningPass and PlanLoweringPass so both resolve
// the DSL's symbolic argument tokens ("ident", "%gtid", "%ident:<flag>", ...)
// the same way.  Internal to lib/Transforms.

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
namespace omp_lower {

// --- Type helpers -----------------------------------------------------------
Type ptrTy(MLIRContext *ctx);
Type i32Ty(MLIRContext *ctx);

// --- DSL-owned ABI layouts --------------------------------------------------
// Expand a "%struct:t0,t1,..." property into an LLVM literal struct type.
// Absent or malformed falls back to the caller's default.
LLVM::LLVMStructType parseStructProp(MLIRContext *ctx, llvm::StringRef prop,
                                     LLVM::LLVMStructType fallback);

// Map a DSL ABI type name ("i8", "i32", "i64", "ptr") to an MLIR type; an
// empty or unknown name keeps the caller's default.
Type parseAbiTypeProp(MLIRContext *ctx, llvm::StringRef name, Type fallback);

// --- Runtime declarations ---------------------------------------------------
// Get, creating on first use, a private external decl for a runtime entry point.
func::FuncOp getOrInsertDecl(ModuleOp module, llvm::StringRef name,
                             ArrayRef<Type> argTypes, OpBuilder &builder);

// Same, for an entry point returning a value.
func::FuncOp getOrInsertDeclWithReturn(ModuleOp module, llvm::StringRef name,
                                       ArrayRef<Type> argTypes, Type returnType,
                                       OpBuilder &builder);

// --- ident_t ----------------------------------------------------------------
// KMP_IDENT_KMPC, always set.  Idents with exactly these flags are the
// "default ident" that call sites cache.
constexpr uint32_t kIdentKmpc = 0x02;

// Values match clang's OpenMPLocationFlags (CGOpenMPRuntime.cpp).
uint32_t identFlagBits(llvm::StringRef tok);

// Recognise "ident", "%ident", or "%ident:<flag>"; on match set `flagsOut`.
bool parseIdentRef(llvm::StringRef s, uint32_t &flagsOut);

// One private constant global per distinct flags value, sharing one psource.
Value getOrCreateIdent(ModuleOp module, OpBuilder &builder, Location loc,
                       MLIRContext *ctx, uint32_t flags);

// Default (KMPC) flags reuse the caller's cached ident; others get their own.
Value resolveIdentToken(uint32_t flags, ModuleOp module, OpBuilder &builder,
                        Location loc, MLIRContext *ctx,
                        llvm::function_ref<Value()> defaultIdent);

// Normalise a clause value to i1 for use as a cond_br condition.
Value clauseToI1(OpBuilder &builder, Location loc, Value v);

// --- proc_bind --------------------------------------------------------------
// Map a proc_bind kind by name to the runtimes' value; nullopt if unknown.
std::optional<uint32_t> procBindEnumValue(llvm::StringRef kind);

// --- Symbolic token resolution ----------------------------------------------
// Resolve a DSL argument token, in order: a site-specific binding (so a
// construct's own tokens and `let` results shadow the built-ins), an ident
// reference, "%gtid", else an undef ptr.  resolveIdent/resolveGtid are seams
// so each caller keeps its own caching and lazy-materialisation policy.
// Literal arguments stay in each caller's arg loop, which types them.
Value resolveSymbolToken(llvm::StringRef s, OpBuilder &builder, Location loc,
                         const llvm::StringMap<Value> &bindings,
                         llvm::function_ref<Value(uint32_t)> resolveIdent,
                         llvm::function_ref<Value()> resolveGtid);

// No-bindings overload, for sites whose only tokens are ident/%gtid.
Value resolveSymbolToken(llvm::StringRef s, OpBuilder &builder, Location loc,
                         llvm::function_ref<Value(uint32_t)> resolveIdent,
                         llvm::function_ref<Value()> resolveGtid);

} // namespace omp_lower
} // namespace mlir
