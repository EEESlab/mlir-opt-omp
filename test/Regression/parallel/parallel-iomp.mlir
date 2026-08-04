// An omp.parallel becomes an omp_lower.construct whose invoke plan forks the
// runtime via __kmpc_fork_call; the parallel body is moved into the construct's
// region (still un-outlined at this stage).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower | FileCheck %s

func.func @empty_parallel() {
  omp.parallel {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @empty_parallel
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "iomp"
// CHECK-SAME:    construct = "parallel"
// CHECK-SAME:    __kmpc_fork_call
