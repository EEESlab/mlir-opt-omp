// The libgomp wsloop has no runtime loop call at all — `emit thread_bounds`
// computes each thread's slice inline — so `GOMP_barrier` in the post block is
// the only thing `nowait` gates.
//
// @sink after the loop keeps the barrier from being the region's trailing op,
// so --omp-barrier-elim could not drop it as redundant with the fork's implicit
// join, which would make the nowait case pass for the wrong reason.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @sink(i32)

func.func @wsloop_nowait() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop nowait {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    llvm.call @sink(%ub) : (i32) -> ()
    omp.terminator
  }
  return
}

func.func @wsloop_wait() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    llvm.call @sink(%ub) : (i32) -> ()
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func {{.*}}@outlined_parallel_0
// CHECK-NOT:     call @GOMP_barrier
// CHECK:         llvm.call @sink

// CHECK-LABEL: func.func {{.*}}@outlined_parallel_1
// CHECK:         call @GOMP_barrier
// CHECK:         llvm.call @sink
