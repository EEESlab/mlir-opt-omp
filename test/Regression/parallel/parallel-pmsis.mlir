// The same input under the pmsis runtime forks the cluster team via
// ext_pi_cl_team_fork (a shim over the PMSIS API, see
// quick-compile/pulp/interface-adapter.c) instead of a __kmpc_/GOMP_ call.
// The team size is fixed at 8 in rules.dsl, not taken from a clause.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower | FileCheck %s

func.func @empty_parallel() {
  omp.parallel {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @empty_parallel
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "pmsis"
// CHECK-SAME:    construct = "parallel"
// CHECK-SAME:    ext_pi_cl_team_fork
