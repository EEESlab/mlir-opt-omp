// An omp.task becomes an omp_lower.construct whose invoke plan schedules the
// task via GOMP_task; the task body is moved into the construct's region
// (still un-outlined at this stage).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower | FileCheck %s

func.func @empty_task() {
  omp.task {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @empty_task
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "libgomp"
// CHECK-SAME:    construct = "task"
// CHECK-SAME:    GOMP_task
