// An explicit schedule(static) under pmsis.  Unlike iomp and libgomp, the pmsis
// wsloop construct in rules.dsl carries NO `when schedule == static` guard, so
// it matches whatever schedule is written — which is why schedule(dynamic) is
// silently mislowered here (schedule-dynamic-pmsis.mlir).
//
// That makes this test worth having anyway, and worth reading carefully: it
// pins that static gets the block distribution it is supposed to get, without
// implying the guard exists.  If the guard is ever added — the right fix for
// the dynamic gap — this test is what says static must keep matching.
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
