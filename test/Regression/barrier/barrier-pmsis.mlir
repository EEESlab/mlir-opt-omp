// The same input under the pmsis runtime selects ext_pi_cl_team_barrier.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower | FileCheck %s

func.func @barrier_only() {
  omp.barrier
  return
}

// CHECK-LABEL: func.func @barrier_only
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "pmsis"
// CHECK-SAME:    construct = "barrier"
// CHECK-SAME:    ext_pi_cl_team_barrier
