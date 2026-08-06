// The same input under the pmsis runtime selects ext_pi_cl_team_barrier.
//
// Two runs, for the reason spelled out in barrier-libgomp.mlir: the first reads
// the call name out of the plan attribute, the second checks it is actually
// emitted.  The ext_pi_cl_team_barrier in nowait-pmsis.mlir and wsloop-pmsis.mlir
// comes from the wsloop `post` block, so without the second run nothing covers
// the barrier construct's own emission.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower | FileCheck %s
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan \
// RUN:   | FileCheck %s --check-prefix=LOWERED

func.func @barrier_only() {
  omp.barrier
  return
}

// The plan the rules produced:
// CHECK-LABEL: func.func @barrier_only
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "pmsis"
// CHECK-SAME:    construct = "barrier"
// CHECK-SAME:    ext_pi_cl_team_barrier

// What actually came out — no arguments, and the construct consumed.
// LOWERED-LABEL: func.func @barrier_only
// LOWERED:         call @ext_pi_cl_team_barrier() : () -> ()
// LOWERED-NOT:     omp_lower.construct
