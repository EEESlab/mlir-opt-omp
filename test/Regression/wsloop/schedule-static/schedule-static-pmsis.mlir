// An explicit schedule(static) under pmsis.  Like iomp and libgomp, the pmsis
// wsloop construct in rules.dsl is guarded by `when schedule == static`, so
// this is the side of that guard which must keep matching — the other side,
// where nothing matches and the evaluation fails, is
// schedule-dynamic-unsupported-pmsis.mlir.
//
// What the guard selects here is the inline block distribution, not a runtime
// call: pmsis asks for `emit thread_bounds`, so the CHECKs below are the core
// id and team size the bounds are computed from.
//
// @sink after the loop keeps the barrier from being the region's trailing op,
// so --omp-barrier-elim could not drop it as redundant with the fork's join.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
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
// CHECK-DAG:     call @ext_pi_core_id
// CHECK-DAG:     call @ext_pi_cl_nb_cores
// CHECK-NOT:     __kmpc_for_static_init
// CHECK-NOT:     GOMP_loop
// CHECK:         call @ext_pi_cl_team_barrier
// CHECK:         llvm.call @sink
