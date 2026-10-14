// The implicit barrier of a work-sharing loop that ends a parallel region.
// The team join synchronises anyway, so the loop's own barrier is redundant.
// The pass drops it by setting `nowait`, the condition every runtime's wsloop
// rule already guards its barrier call with, so rules.dsl needs no change.
//
// The two functions are the two shapes PolyBench-OMP has.  gemm is one loop in
// a region, always removable.  atax is three: the first two barriers separate
// loops that read what the previous one wrote and must stay, only the last
// goes.  Keeping both here makes the CHECKs a statement about *position*
// rather than about wsloops in general.
//
// No runtime is selected — the pass reads no DSL — so one RUN line covers all
// three.  no-barrier-call-*.mlir carries this through to the emitted call.
//
// RUN: mlir-opt-omp %s --omp-barrier-elim | FileCheck %s

llvm.func @work(!llvm.ptr)

func.func @gemm_shape(%p: !llvm.ptr, %lb: i32, %ub: i32, %st: i32) {
  omp.parallel {
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

func.func @atax_shape(%p: !llvm.ptr, %lb: i32, %ub: i32, %st: i32) {
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

// The lone loop of the gemm shape ends the region, so it becomes nowait.
// CHECK-LABEL: func.func @gemm_shape
// CHECK:         omp.wsloop nowait

// Of the three atax loops only the last one does.  Matching `omp.wsloop {` for
// the first two is what pins that they were left alone: a `nowait` one prints
// as `omp.wsloop nowait {` and would not match.
// CHECK-LABEL: func.func @atax_shape
// CHECK:         omp.wsloop {
// CHECK:         omp.wsloop {
// CHECK:         omp.wsloop nowait
