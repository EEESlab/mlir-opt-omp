// Removes team barriers the surrounding structure already guarantees: the one
// that closes a parallel region, the one that opens it, one following another
// barrier, and the implicit barrier of a wsloop that ends a region.
//
// It runs on the omp dialect, before --omp-to-omp-lower, and that is the point:
// after lowering a barrier is an opaque runtime call LLVM can never remove.
// Stating the rule on the dialect also serves all three runtimes at once.  An
// implicit barrier is dropped by setting `nowait`, the flag every runtime's
// wsloop rule already keys its closing call on.
//
// ASSUMES a parallel region joins its team on the way out — needed by the two
// rules that drop a barrier at the end of a region.  __kmpc_fork_call and
// GOMP_parallel do join, and no runtime gives `parallel` a `post` block.  For
// pmsis it rests on pi_cl_team_fork being synchronous: unconfirmed, and pmsis
// is where a removed barrier is worth the most.

#include "OmpLowering/Transforms/OmpBarrierElimPass.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// Ops the searches below may skip over: only one that touches memory can be the
// reason a barrier is needed.  isMemoryEffectFree answers no when it cannot
// tell, which is the safe way round.
bool isTransparent(Operation *op) { return isMemoryEffectFree(op); }

// The omp.parallel whose body block holds `op` directly, or null.  Nesting in
// anything else gives null, which stops every rule below from reasoning across
// a boundary it cannot see through.
omp::ParallelOp parentParallel(Operation *op) {
  Block *block = op->getBlock();
  if (!block) return nullptr;
  return llvm::dyn_cast_or_null<omp::ParallelOp>(block->getParentOp());
}

// Nothing meaningful separates `op` from the end of its parallel region, so the
// team join is the next synchronisation.  It must be an omp.terminator: a
// multi-block body ends its first block with a branch, and taking that for the
// end of the region would mean reading past a jump.
bool endsParallelRegion(Operation *op) {
  if (!parentParallel(op)) return false;
  for (Operation *next = op->getNextNode(); next; next = next->getNextNode()) {
    if (llvm::isa<omp::TerminatorOp>(next)) return true;
    if (!isTransparent(next)) return false;
  }
  return false;
}

// The mirror: nothing meaningful runs before `op`, so there is nothing yet to
// synchronise.
bool startsParallelRegion(Operation *op) {
  if (!parentParallel(op)) return false;
  for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode())
    if (!isTransparent(prev)) return false;
  return true;
}

Operation *previousMeaningful(Operation *op) {
  for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode())
    if (!isTransparent(prev)) return prev;
  return nullptr;
}

bool leavesImplicitBarrier(omp::WsloopOp op) { return !op.getNowait(); }

// Clauses needing the loop's own barrier whatever surrounds it.  Neither is in
// scope here, but getting it wrong would be silent, so both are checked.
bool needsOwnBarrier(omp::WsloopOp op) {
  return !op.getReductionVars().empty() || op.getPrivateNeedsBarrier();
}

struct OmpBarrierElimPass
    : public PassWrapper<OmpBarrierElimPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OmpBarrierElimPass)

  OmpBarrierElimPass() = default;
  // Registration clones the pass, and a Statistic holds an atomic counter it
  // cannot copy, so the copy starts its own counters.
  OmpBarrierElimPass(const OmpBarrierElimPass &other) : PassWrapper(other) {}

  llvm::StringRef getArgument() const override { return "omp-barrier-elim"; }
  llvm::StringRef getDescription() const override {
    return "Remove team barriers the surrounding OpenMP structure guarantees";
  }

  Pass::Statistic explicitRemoved{
      this, "explicit-barriers-removed",
      "omp.barrier operations erased as redundant"};
  Pass::Statistic implicitRemoved{
      this, "implicit-barriers-removed",
      "omp.wsloop implicit barriers dropped by setting nowait"};

  // One sweep of all four rules.  They feed each other, so runOnOperation
  // repeats until a sweep is quiet.
  bool sweep(ModuleOp module) {
    bool changed = false;

    // Explicit barriers first: erasing one can expose a wsloop to the rule
    // below, never the other way round, so this order converges sooner.
    SmallVector<omp::BarrierOp> deadBarriers;
    module.walk([&](omp::BarrierOp op) {
      if (endsParallelRegion(op)) {
        deadBarriers.push_back(op);
        return;
      }
      if (startsParallelRegion(op)) {
        deadBarriers.push_back(op);
        return;
      }
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

    // The rule that pays: `parallel { for }` is the common shape, and there the
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
