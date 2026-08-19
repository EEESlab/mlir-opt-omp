// Which barrier goes, at the level of the emitted call.  Two work-sharing
// loops in one region: the first loop's barrier separates them and is
// load-bearing, only the trailing one is redundant with the team join.
//
// wsloop-trailing.mlir makes the same statement on the omp dialect, where it
// is a `nowait` placement; this carries it through to __kmpc_barrier, which is
// what actually costs cycles.  no-barrier-call-*.mlir covers the single-loop
// shape on all three runtimes, so only the two-loop one is here — it is the
// case that pins *position* rather than the removal as such.
//
// Two RUN lines over one input: the baseline pins that both calls were there
// to begin with, so the second is not passing merely because barriers stopped
// being emitted.  Until this rule moved into --omp-barrier-elim it lived in
// OmpOutliningPass, and the baseline below could not have shown a barrier.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan \
// RUN:   | FileCheck %s --check-prefix=BASELINE
// RUN: mlir-opt-omp %s --omp-barrier-elim --omp-lower-dsl=%rules_dsl \
// RUN:   --omp-lower-runtime=iomp --omp-to-omp-lower --omp-outline \
// RUN:   --omp-lower-plan | FileCheck %s --check-prefix=ELIDED

llvm.func @work(!llvm.ptr)

func.func @parallel_two_loops(%p: !llvm.ptr, %lb: i32, %ub: i32, %st: i32) {
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : i32 = (%lb) to (%ub) step (%st) {
        llvm.call @work(%p) : (!llvm.ptr) -> ()
        omp.yield
      }
    }
    omp.wsloop {
      omp.loop_nest (%i) : i32 = (%lb) to (%ub) step (%st) {
        llvm.call @work(%p) : (!llvm.ptr) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// Without the pass both loops emit their own barrier.
// BASELINE-LABEL: func.func {{.*}}@outlined_parallel_
// BASELINE:       call @__kmpc_barrier
// BASELINE:       call @__kmpc_barrier
// BASELINE:       return

// With it exactly one survives — the one separating the two loops.  The
// CHECK-NOT between the two matches is what makes this a count and not just
// "at least one".
// ELIDED-LABEL: func.func {{.*}}@outlined_parallel_
// ELIDED:       call @__kmpc_barrier
// ELIDED-NOT:   call @__kmpc_barrier
// ELIDED:       return
