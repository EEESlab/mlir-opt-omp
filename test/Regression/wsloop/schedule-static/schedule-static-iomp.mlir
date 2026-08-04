// An EXPLICIT schedule(static) must select the same construct as the implicit
// default.  This is not tautological: rules.dsl guards the construct with
// `when schedule == static`, while the pass fills the context either with the
// literal default "static" or with omp::stringifyClauseScheduleKind of the
// clause — so an explicit clause exercises the stringified spelling, and a
// mismatch there would silently fall through to "no matching construct".
//
// 34 is the kmp_sch_static_greedy schedule constant; the trailing 1 is
// `default_chunk` from the runtime block of rules.dsl.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
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

// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK:         call @__kmpc_for_static_init_4
// CHECK:         call @__kmpc_for_static_fini
// CHECK:         call @__kmpc_barrier
// CHECK:         llvm.call @sink
