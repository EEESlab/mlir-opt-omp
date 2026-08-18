// Explicit omp.barrier operations that the surrounding structure already
// guarantees, and the ones it does not.
//
// @removable holds all the removable shapes at once, deliberately: the rules
// feed each other, and only running them together tests that.  Erasing the
// adjacent barrier leaves the wsloop next to the trailing one; erasing that
// leaves the wsloop ending the region, which is what makes it nowait.
//
// RUN: mlir-opt-omp %s --omp-barrier-elim | FileCheck %s

llvm.func @work(!llvm.ptr)

func.func @removable(%p: !llvm.ptr, %lb: i32, %ub: i32, %st: i32) {
  omp.parallel {
    omp.barrier                                     // opens the region: the fork just synchronised
    llvm.call @work(%p) : (!llvm.ptr) -> ()
    omp.barrier                                     // stays: work above, a loop below
    omp.barrier                                     // follows a barrier with nothing in between
    omp.wsloop {
      omp.loop_nest (%i) : i32 = (%lb) to (%ub) step (%st) {
        llvm.call @work(%p) : (!llvm.ptr) -> ()
        omp.yield
      }
    }
    omp.barrier                                     // closes the region: the join will synchronise
    omp.terminator
  }
  return
}

// A barrier with work on both sides is doing its job.
func.func @kept(%p: !llvm.ptr) {
  omp.parallel {
    llvm.call @work(%p) : (!llvm.ptr) -> ()
    omp.barrier
    llvm.call @work(%p) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// Outside a parallel region there is no team join to lean on, and no fork
// above: the pass must not touch it.
func.func @toplevel() {
  omp.barrier
  return
}

// Exactly one barrier survives, the one between the call and the loop, and the
// loop ends up nowait.  CHECK-NOT between the label and the call rules out a
// surviving leading barrier; the one between the loop and the next label rules
// out a surviving trailing one.
// CHECK-LABEL: func.func @removable
// CHECK-NOT:     omp.barrier
// CHECK:         llvm.call @work
// CHECK:         omp.barrier
// CHECK-NOT:     omp.barrier
// CHECK:         omp.wsloop nowait
// CHECK-NOT:     omp.barrier

// CHECK-LABEL: func.func @kept
// CHECK:         llvm.call @work
// CHECK:         omp.barrier
// CHECK:         llvm.call @work

// CHECK-LABEL: func.func @toplevel
// CHECK:         omp.barrier
