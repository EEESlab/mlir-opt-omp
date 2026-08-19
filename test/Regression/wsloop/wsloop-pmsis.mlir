// The pmsis wsloop is the one work-sharing path with no runtime loop API behind
// it: rules.dsl asks for `emit thread_bounds`, so the pass computes each core's
// slice inline from ext_pi_core_id / ext_pi_cl_nb_cores (the DIVMOD block
// distribution) rather than calling a __kmpc_for_static_init_4 equivalent.
// The `post` block then closes the loop with a team barrier, since the wsloop
// is not nowait.
//
// The @sink call after the loop keeps that barrier from being the region's
// trailing op — otherwise --omp-barrier-elim would drop it as redundant with
// the fork's implicit join (see barrier-elim/wsloop-trailing.mlir).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @sink()

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
    llvm.call @sink() : () -> ()
    omp.terminator
  }
  return
}

// The bounds come from the core id and the team size, in either order.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-DAG:     call @ext_pi_core_id
// CHECK-DAG:     call @ext_pi_cl_nb_cores
// No runtime work-sharing call: the distribution is computed inline.
// CHECK-NOT:     __kmpc_for_static_init
// CHECK-NOT:     GOMP_loop
// The trailing barrier survives because @sink follows it.
// CHECK:         call @ext_pi_cl_team_barrier
