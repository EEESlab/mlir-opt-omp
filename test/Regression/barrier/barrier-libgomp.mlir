// The same input under the libgomp runtime selects GOMP_barrier instead.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower | FileCheck %s

func.func @barrier_only() {
  omp.barrier
  return
}

// CHECK-LABEL: func.func @barrier_only
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "libgomp"
// CHECK-SAME:    construct = "barrier"
// CHECK-SAME:    GOMP_barrier
