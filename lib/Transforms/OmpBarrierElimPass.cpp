// OmpBarrierElimPass.cpp
//
// Removes team barriers the surrounding structure already guarantees.  Four
// shapes: the barrier that closes a parallel region, the one that opens it,
// one that follows another barrier, and the implicit barrier of a work-sharing
// loop that ends a parallel region.
//
// It runs on the omp dialect, before --omp-to-omp-lower, and that is the point.
// After lowering a barrier is a call to __kmpc_barrier, GOMP_barrier or
// ext_pi_cl_team_barrier — opaque to LLVM, which records nothing about what it
// synchronises and so can never remove it.  Here the structure is still
// readable; and since the rule is spelled on the dialect rather than in a
// runtime's rules, all three runtimes get it at once.
//
// An implicit barrier is never emitted directly: it is removed by setting
// `nowait` on the omp.wsloop, which is the condition every runtime's wsloop
// rule already guards its barrier call with (`when not nowait => call ...`).
//
// ASSUMES a parallel region joins its team on the way out: the fork call does
// not return until every thread has finished.  The two rules that drop a
// barrier at the end of a region need this, the wsloop one among them.  The
// other two stand on their own.
//
// __kmpc_fork_call and GOMP_parallel return only after the join, and rules.dsl
// agrees: no runtime gives `parallel` a `post` block, which is where an
// exit-synchronising call would have to go.  For pmsis it rests on
// pi_cl_team_fork being synchronous — unconfirmed, and pmsis is where a
// removed barrier is worth the most.

#include "OmpLowering/Transforms/OmpBarrierElimPass.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// Ops the searches below may skip over.  A barrier makes one thread's writes
// visible to the others, so only an op that touches memory can be the reason
// one is needed.  isMemoryEffectFree answers no when it cannot tell, which is
// the safe way round: it keeps a barrier rather than dropping it.
bool isTransparent(Operation *op) { return isMemoryEffectFree(op); }

// The omp.parallel whose body block holds `op` directly, or null.  Nesting
// inside anything else — an scf.if, a wsloop, another parallel — gives null,
// which stops every rule below from reasoning across a boundary it cannot see
// through.
omp::ParallelOp parentParallel(Operation *op) {
  Block *block = op->getBlock();
  if (!block) return nullptr;
  return llvm::dyn_cast_or_null<omp::ParallelOp>(block->getParentOp());
}

// Nothing meaningful separates `op` from the end of its parallel region, so
// the team join is the next synchronisation.
//
// It must be an omp.terminator: a multi-block body ends its first block with a
// branch, which is a terminator too, and taking that for the end of the region
// would mean reading past a jump into code never looked at.
bool endsParallelRegion(Operation *op) {
  if (!parentParallel(op)) return false;
  for (Operation *next = op->getNextNode(); next; next = next->getNextNode()) {
    if (llvm::isa<omp::TerminatorOp>(next)) return true;
    if (!isTransparent(next)) return false;
  }
  return false;
}

// The mirror: nothing meaningful runs between the start of the region and
// `op`, so there is nothing yet for a barrier to synchronise.
bool startsParallelRegion(Operation *op) {
  if (!parentParallel(op)) return false;
  for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode())
    if (!isTransparent(prev)) return false;
  return true;
}

// The last op before `op` that does anything, or null.
Operation *previousMeaningful(Operation *op) {
  for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode())
    if (!isTransparent(prev)) return prev;
  return nullptr;
}

bool leavesImplicitBarrier(omp::WsloopOp op) { return !op.getNowait(); }

// Clauses that need the loop's own barrier whatever surrounds it: a reduction
// combines partial results after the last iteration, private_needs_barrier
// asks for one outright.  Neither is in scope here, but getting this wrong
// would be silent, so they are checked rather than assumed absent.
bool needsOwnBarrier(omp::WsloopOp op) {
  return !op.getReductionVars().empty() || op.getPrivateNeedsBarrier();
}

struct OmpBarrierElimPass
    : public PassWrapper<OmpBarrierElimPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OmpBarrierElimPass)

  OmpBarrierElimPass() = default;
  // Registration clones the pass, and a Statistic holds an atomic counter it
  // cannot copy, so the copy starts its own counters rather than defaulting.
  OmpBarrierElimPass(const OmpBarrierElimPass &other) : PassWrapper(other) {}

  llvm::StringRef getArgument() const override { return "omp-barrier-elim"; }
  llvm::StringRef getDescription() const override {
    return "Remove team barriers the surrounding OpenMP structure guarantees";
  }

  // Reported by --mlir-pass-statistics: how the evaluation counts removals
  // without a separate tool.
  Pass::Statistic explicitRemoved{
      this, "explicit-barriers-removed",
      "omp.barrier operations erased as redundant"};
  Pass::Statistic implicitRemoved{
      this, "implicit-barriers-removed",
      "omp.wsloop implicit barriers dropped by setting nowait"};

  // One sweep of all four rules, reporting whether anything changed.  The
  // rules feed each other — erasing a trailing omp.barrier leaves the wsloop
  // before it ending the region — so runOnOperation repeats until a sweep is
  // quiet.
  bool sweep(ModuleOp module) {
    bool changed = false;

    // Explicit barriers first: erasing one can expose a wsloop to the rule
    // below, never the other way round, so this order converges sooner.
    SmallVector<omp::BarrierOp> deadBarriers;
    module.walk([&](omp::BarrierOp op) {
      // Closes the region: the team join synchronises anyway.
      if (endsParallelRegion(op)) {
        deadBarriers.push_back(op);
        return;
      }
      // Opens it: nothing has happened yet.
      if (startsParallelRegion(op)) {
        deadBarriers.push_back(op);
        return;
      }
      // Follows another synchronisation with nothing in between — a second
      // barrier, or a wsloop that ends in one of its own.
      if (Operation *prev = previousMeaningful(op)) {
        if (llvm::isa<omp::BarrierOp>(prev)) {
          deadBarriers.push_back(op);
          return;
        }
        if (auto ws = llvm::dyn_cast<omp::WsloopOp>(prev))
          if (leavesImplicitBarrier(ws))
            deadBarriers.push_back(op);
      }
    });
    for (auto op : deadBarriers) {
      op.erase();
      ++explicitRemoved;
      changed = true;
    }

    // The implicit barrier of a wsloop ending a parallel region.  This is the
    // rule that pays: `parallel { for }` is the common shape, and there the
    // loop's barrier and the team join sit back to back.
    module.walk([&](omp::WsloopOp op) {
      if (!leavesImplicitBarrier(op) || needsOwnBarrier(op)) return;
      if (!endsParallelRegion(op)) return;
      op.setNowait(true);
      ++implicitRemoved;
      changed = true;
    });

    return changed;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Each sweep strictly removes barriers, and there are finitely many, so
    // this terminates.
    while (sweep(module)) {}
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> mlir::createOmpBarrierElimPass() {
  return std::make_unique<OmpBarrierElimPass>();
}

void mlir::registerOmpBarrierElimPass() {
  PassRegistration<OmpBarrierElimPass>();
}
