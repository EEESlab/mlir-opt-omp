// The elimination carried through to what libgomp actually emits.  See
// no-barrier-call-iomp.mlir for what the two RUN lines are for.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan \
// RUN:   | FileCheck %s --check-prefix=BASELINE
// RUN: mlir-opt-omp %s --omp-barrier-elim --omp-lower-dsl=%rules_dsl \
// RUN:   --omp-lower-runtime=libgomp --omp-to-omp-lower --omp-outline \
// RUN:   --omp-lower-plan | FileCheck %s --check-prefix=ELIDED

llvm.func @work(!llvm.ptr)

func.func @one_loop(%p: !llvm.ptr, %lb: i32, %ub: i32, %st: i32) {
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

// BASELINE: GOMP_barrier
// ELIDED-NOT: GOMP_barrier
