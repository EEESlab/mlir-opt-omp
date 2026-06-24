// Full lowering of an omp.task under libgomp: the body is outlined into a
// closure (void(ptr)) and the call site schedules it via GOMP_task, passing
// the capture struct together with its size/alignment and the if-clause.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @task_if(%arg0: !llvm.ptr, %cond: i1) {
  omp.task if(%cond) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// The body is outlined into a closure taking a single data pointer.
// CHECK: func.func {{.*}}@outlined_task_0(%{{.*}}: !llvm.ptr)

// The if-clause (i1) is widened to i8 (C _Bool) and the task is scheduled.
// CHECK-LABEL: func.func @task_if
// CHECK:       llvm.zext %{{.*}} : i1 to i8
// CHECK:       call @GOMP_task
