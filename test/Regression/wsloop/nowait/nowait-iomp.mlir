// `nowait` drops the implicit barrier at the end of a work-sharing loop.  In
// rules.dsl the iomp post block is `__kmpc_for_static_fini` unconditionally plus
// `when not nowait => __kmpc_barrier`, so nowait must remove the barrier and
// keep the fini.
//
// Both regions call @sink after the loop on purpose: a trailing team barrier is
// elided anyway as redundant with the fork's implicit join (see
// barrier-elision-iomp.mlir), which would make the nowait case pass for the
// wrong reason.  With @sink in the way, the barrier below is present only if
// `not nowait` really gated it.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
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

// nowait: fini stays, barrier goes.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_0
// CHECK:         call @__kmpc_for_static_fini
// CHECK-NOT:     call @__kmpc_barrier
// CHECK:         llvm.call @sink

// default: both are emitted.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_1
// CHECK:         call @__kmpc_for_static_fini
// CHECK:         call @__kmpc_barrier
// CHECK:         llvm.call @sink
