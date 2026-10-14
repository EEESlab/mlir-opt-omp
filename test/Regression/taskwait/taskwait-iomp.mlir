// A bare omp.taskwait lowers to an omp_lower.construct carrying the iomp
// __kmpc_omp_taskwait call (leaf construct, like barrier).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower | FileCheck %s

func.func @taskwait_only() {
  omp.taskwait
  return
}

// CHECK-LABEL: func.func @taskwait_only
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "iomp"
// CHECK-SAME:    construct = "taskwait"
// CHECK-SAME:    __kmpc_omp_taskwait
