// A task nested inside a parallel region (the realistic `parallel { task }`
// pattern).  The parallel is outlined first; the nested task becomes a
// GOMP_task call *inside* the parallel's outlined function, with its own body
// outlined into a second closure.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @nested(%arg0: !llvm.ptr) {
  omp.parallel {
    omp.task {
      llvm.call @use(%arg0) : (!llvm.ptr) -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}

// The task body is outlined into its own closure...
// CHECK: func.func {{.*}}@outlined_task_{{[0-9]+}}(%{{.*}}: !llvm.ptr)
// CHECK:   llvm.call @use

// ...and the parallel's outlined function schedules it via GOMP_task.
// The call carries all 10 GOMP_task params (GCC 11+ ABI); trailing !llvm.ptr
// is `detach`. A 9-arg call would be an ABI mismatch.
// CHECK: func.func {{.*}}@outlined_parallel_{{[0-9]+}}(%{{.*}}: !llvm.ptr)
// CHECK:   call @GOMP_task({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i8, i32, !llvm.ptr, i32, !llvm.ptr) -> ()

// The original function forks the parallel region.
// CHECK-LABEL: func.func @nested
// CHECK:       call @GOMP_parallel
