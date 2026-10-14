// The elimination carried through to what iomp actually emits.  The pass only
// sets `nowait` on the omp dialect; that it reaches the runtime depends on the
// wsloop rule guarding its barrier call with `when not nowait`.
//
// Two RUN lines over one input: the first shows the call is there without the
// pass, the second that it is gone with it.  Checking only the second would
// pass just as well if the runtime had stopped emitting barriers entirely.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan \
// RUN:   | FileCheck %s --check-prefix=BASELINE
// RUN: mlir-opt-omp %s --omp-barrier-elim --omp-lower-dsl=%rules_dsl \
// RUN:   --omp-lower-runtime=iomp --omp-to-omp-lower --omp-outline \
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

// BASELINE: __kmpc_barrier
// ELIDED-NOT: __kmpc_barrier
