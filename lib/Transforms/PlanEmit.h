// PlanEmit.h
//
// The vocabulary shared by the two passes that turn a LoweringPlan into calls:
// runtime declarations, ident_t globals, and the resolution of the DSL's
// symbolic argument tokens ("%gtid", "ident", "%ident:<flag>", ...) to SSA
// values.
//
// It exists so OmpOutliningPass and PlanLoweringPass resolve those tokens the
// same way.  They used to disagree: the outlining pass knew the vocabulary,
// the plan pass resolved everything it did not recognise to an undef pointer,
// which is why top-level leaf constructs had to be lowered in the outlining
// pass to avoid an undef gtid.
//
// Internal to lib/Transforms — not part of the public OmpLowering headers.

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
// Expand a DSL struct-layout property of the form "%struct:t0,t1,..." into an
// LLVM literal struct type.  An absent or malformed property falls back to the
// caller's default so older DSL files keep working.  Shared because both passes
// read the same layout: the outlining pass builds the entry prolog from it, the
// plan pass reaches task->shareds through it.
LLVM::LLVMStructType parseStructProp(MLIRContext *ctx, llvm::StringRef prop,
                                     LLVM::LLVMStructType fallback);

// Map a DSL ABI type name ("i8", "i32", "i64", "ptr") to an MLIR type.  An
// empty or unknown name falls back to the caller's default, which is what the
// properties naming a runtime's widths rely on (chunk_index, chunk_result):
// a DSL file that says nothing keeps the pass's own choice.
Type parseAbiTypeProp(MLIRContext *ctx, llvm::StringRef name, Type fallback);

// --- Runtime declarations ---------------------------------------------------
// Get (creating on first use) a private external func declaration for a runtime
// entry point, with the given argument types and no result.
func::FuncOp getOrInsertDecl(ModuleOp module, llvm::StringRef name,
                             ArrayRef<Type> argTypes, OpBuilder &builder);

// Same, for an entry point returning a value (e.g. __kmpc_global_thread_num).
func::FuncOp getOrInsertDeclWithReturn(ModuleOp module, llvm::StringRef name,
                                       ArrayRef<Type> argTypes, Type returnType,
                                       OpBuilder &builder);

// --- ident_t ----------------------------------------------------------------
// KMP_IDENT_KMPC: the default ident flag, always set, mirroring
// OMPIRBuilder::getOrCreateIdent.  Idents with exactly these flags are the
// "default ident" that fork/gtid calls seed off and that call sites cache.
constexpr uint32_t kIdentKmpc = 0x02;

// Map a DSL ident flag token to the effective `ident_t.flags` bitmask.
// Values match clang's OpenMPLocationFlags (CGOpenMPRuntime.cpp).
uint32_t identFlagBits(llvm::StringRef tok);

// Recognise a symbolic ident reference: "ident", "%ident", or "%ident:<flag>".
// On match, set `flagsOut` to the effective flag bits and return true.
bool parseIdentRef(llvm::StringRef s, uint32_t &flagsOut);

// Get (creating on first use) the address of the `ident_t` global for a given
// flags value.  One private constant global per distinct flags value, all
// sharing a single default psource string.
Value getOrCreateIdent(ModuleOp module, OpBuilder &builder, Location loc,
                       MLIRContext *ctx, uint32_t flags);

// Shared policy for the resolveIdent seam: default (KMPC) flags reuse the
// caller's cached default ident; any other flags get their own global.
Value resolveIdentToken(uint32_t flags, ModuleOp module, OpBuilder &builder,
                        Location loc, MLIRContext *ctx,
                        llvm::function_ref<Value()> defaultIdent);

// Normalise a value to i1 for use as a cond_br condition / select predicate
// (clause values are typically already i1).
Value clauseToI1(OpBuilder &builder, Location loc, Value v);

// --- proc_bind --------------------------------------------------------------
// Map an OpenMP proc_bind kind, by name, to the value the runtimes give it.
// std::nullopt for a kind this table does not know.
std::optional<uint32_t> procBindEnumValue(llvm::StringRef kind);

// --- Symbolic token resolution ----------------------------------------------
// Resolve a DSL argument token against one shared vocabulary:
//   1. a site-specific binding (looked up first, so a construct's own tokens and
//      `let`-bound results shadow the built-ins);
//   2. an ident reference ("ident"/"%ident"/"%ident:<flag>") -> resolveIdent;
//   3. "%gtid"                                               -> resolveGtid;
//   4. anything unrecognised                                 -> an undef ptr.
// resolveIdent/resolveGtid are seams so each caller keeps its own caching and
// lazy-materialisation policy (e.g. the fork path only emits the gtid call when
// a token actually references it).  Literal (integer/bool) arguments stay in
// each caller's arg loop: most sites emit the same arith i32 constant, but
// wsloop types them as the induction variable (LLVM constant) and the packed
// invoke adds i8-bool / i64-size forms.
Value resolveSymbolToken(llvm::StringRef s, OpBuilder &builder, Location loc,
                         const llvm::StringMap<Value> &bindings,
                         llvm::function_ref<Value(uint32_t)> resolveIdent,
                         llvm::function_ref<Value()> resolveGtid);

// No-bindings overload for sites whose only symbolic tokens are ident/%gtid
// (e.g. barrier).
Value resolveSymbolToken(llvm::StringRef s, OpBuilder &builder, Location loc,
                         llvm::function_ref<Value(uint32_t)> resolveIdent,
                         llvm::function_ref<Value()> resolveGtid);

} // namespace omp_lower
} // namespace mlir
