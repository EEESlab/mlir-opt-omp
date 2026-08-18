// The cases where a work-sharing loop's implicit barrier has to stay.  A wrong
// answer here is a race, not a diagnostic: nothing downstream would catch it.
//
// RUN: mlir-opt-omp %s --omp-barrier-elim | FileCheck %s

llvm.func @work(!llvm.ptr)

// Work follows the loop inside the same region, and it may read what another
// thread's iterations wrote.  The join at the end of the region is too late.
func.func @work_after(%p: !llvm.ptr, %lb: i32, %ub: i32, %st: i32) {
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : i32 = (%lb) to (%ub) step (%st) {
        llvm.call @work(%p) : (!llvm.ptr) -> ()
        omp.yield
      }
    }
    llvm.call @work(%p) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// No enclosing parallel region, so there is no team join to fall back on.
func.func @no_parallel(%p: !llvm.ptr, %lb: i32, %ub: i32, %st: i32) {
  omp.wsloop {
    omp.loop_nest (%i) : i32 = (%lb) to (%ub) step (%st) {
      llvm.call @work(%p) : (!llvm.ptr) -> ()
      omp.yield
    }
  }
  return
}

// Already nowait: nothing to do, and running the pass again must not disturb
// it.  This is what makes the pass safe to run twice in a pipeline.
func.func @already_nowait(%p: !llvm.ptr, %lb: i32, %ub: i32, %st: i32) {
  omp.parallel {
    omp.wsloop nowait {
      omp.loop_nest (%i) : i32 = (%lb) to (%ub) step (%st) {
        llvm.call @work(%p) : (!llvm.ptr) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// Matching `omp.wsloop {` is the assertion: the nowait form prints as
// `omp.wsloop nowait {` and would not match.
// CHECK-LABEL: func.func @work_after
// CHECK:         omp.wsloop {

// CHECK-LABEL: func.func @no_parallel
// CHECK:         omp.wsloop {

// CHECK-LABEL: func.func @already_nowait
// CHECK:         omp.wsloop nowait
