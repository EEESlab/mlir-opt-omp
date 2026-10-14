// The same elimination as no-barrier-call-libgomp.mlir, on the one wsloop
// lowering where `nowait` does not switch a call off.
//
// --omp-barrier-elim removes a work-sharing loop's implicit barrier by setting
// `nowait` on the omp.wsloop, and every other wsloop rule guards a barrier call
// with `when not nowait`, so the flag simply deletes it.  libgomp's dynamic
// schedule keys two different calls on the flag instead: the work-share has to
// be released either way, and only the barrier inside GOMP_loop_end is what
// nowait drops.  So what the elimination produces here is not "one call fewer"
// but a different call, which is what this test pins.
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
    omp.wsloop schedule(dynamic) {
      omp.loop_nest (%i) : i32 = (%lb) to (%ub) step (%st) {
        llvm.call @work(%p) : (!llvm.ptr) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// The loop closes the region, so its barrier is redundant with the fork's join.
// BASELINE: call @GOMP_loop_end()
// ELIDED-NOT: call @GOMP_loop_end()
// ELIDED: call @GOMP_loop_end_nowait()
