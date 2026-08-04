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
// CHECK: func.func {{.*}}@outlined_task_{{[0-9]+}}(%{{.*}}: !llvm.ptr)

// The if-clause (i1) is widened to i8 (C _Bool) and the task is scheduled.
// The call carries all 10 GOMP_task params (GCC 11+ ABI); the trailing
// !llvm.ptr is `detach`. A 9-arg call would be an ABI mismatch (libgomp would
// read an uninitialised 10th argument).
// CHECK-LABEL: func.func @task_if
// CHECK:       llvm.zext %{{.*}} : i1 to i8
// CHECK:       call @GOMP_task({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i8, i32, !llvm.ptr, i32, !llvm.ptr) -> ()
