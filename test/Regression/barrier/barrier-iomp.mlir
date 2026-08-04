// A bare omp.barrier lowers to an omp_lower.construct carrying the iomp
// __kmpc_barrier call.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower | FileCheck %s

func.func @barrier_only() {
  omp.barrier
  return
}

// CHECK-LABEL: func.func @barrier_only
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "iomp"
// CHECK-SAME:    construct = "barrier"
// CHECK-SAME:    __kmpc_barrier
