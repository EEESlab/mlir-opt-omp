// The same input under the pmsis runtime forks the cluster team via
// ext_pi_cl_team_fork (a shim over the PMSIS API, provided by the PULP
// harness) instead of a __kmpc_/GOMP_ call.
// With no num_threads clause the team size is `default_team_size` from
// rules.dsl (see num_threads-pmsis.mlir for the clause taking its place).
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
