// An explicit schedule(static) under libgomp must select the same construct as
// the implicit default.  Not tautological: rules.dsl guards the construct with
// `when schedule == static`, and the pass fills the context either with the
// literal default "static" or with omp::stringifyClauseScheduleKind of the
// clause.  If those two spellings ever diverge the explicit clause stops
// matching and falls through to "no matching construct" — which is what
// schedule-dynamic-unsupported-libgomp.mlir pins from the other side.
//
// libgomp has no runtime work-sharing API here: `emit thread_bounds` makes the
// pass compute each thread's slice inline from omp_get_thread_num /
// omp_get_num_threads, and the post block closes with GOMP_barrier.
//
// @sink after the loop keeps that barrier from being elided as redundant with
// the fork's implicit join (see barrier-elision-iomp.mlir).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @sink(i32)

func.func @wsloop_static() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop schedule(static) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    llvm.call @sink(%ub) : (i32) -> ()
    omp.terminator
  }
  return
}

// The bounds come from the thread id and the team size, in either order.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-DAG:     call @omp_get_thread_num
// CHECK-DAG:     call @omp_get_num_threads
// No runtime loop call: the distribution is inline.
// CHECK-NOT:     __kmpc_for_static_init
// CHECK-NOT:     GOMP_loop
// The implicit barrier survives because @sink follows it.
// CHECK:         call @GOMP_barrier
// CHECK:         llvm.call @sink
