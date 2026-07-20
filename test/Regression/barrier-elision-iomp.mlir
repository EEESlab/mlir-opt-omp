// A team barrier that is the last op before an outlined parallel region returns
// is redundant with the implicit join barrier of __kmpc_fork_call, so
// OmpOutliningPass drops it (the combined `parallel for` case — clang elides
// the work-sharing barrier the same way).  A barrier that separates two
// work-sharing loops inside the same region is load-bearing and preserved.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

// Case 1: the wsloop is the sole/last construct — its implicit barrier is
// elided while the static-loop fini is kept.
func.func @parallel_for() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK:       call @__kmpc_for_static_fini
// CHECK-NOT:   call @__kmpc_barrier
// CHECK:       return

// Case 2: two back-to-back wsloops — the first loop's barrier separates them
// and must survive; only the second (trailing) loop's barrier is elided.
func.func @parallel_two_loops() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// Exactly one barrier survives in this region (the separating one); the
// trailing one is gone.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK:       call @__kmpc_barrier
// CHECK-NOT:   call @__kmpc_barrier
// CHECK:       return
