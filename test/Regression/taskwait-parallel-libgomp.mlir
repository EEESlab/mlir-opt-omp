// An omp.taskwait inside a libgomp parallel rides into the parallel's outlined
// closure and is lowered there (the same in-outlined-fn path as omp.barrier) to
// a no-argument GOMP_taskwait().  Here it follows a spawned task, the common
// `parallel { task; taskwait }` shape.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @parallel_task_taskwait(%arg0: !llvm.ptr) {
  omp.parallel {
    omp.task {
      llvm.call @use(%arg0) : (!llvm.ptr) -> ()
      omp.terminator
    }
    omp.taskwait
    omp.terminator
  }
  return
}

// The task is scheduled and then the taskwait is emitted, both inside the
// parallel's outlined closure.
// CHECK: func.func {{.*}}@outlined_parallel_{{[0-9]+}}(%{{.*}}: !llvm.ptr)
// CHECK:   call @GOMP_task
// CHECK:   call @GOMP_taskwait() : () -> ()

// The original function forks the parallel region.
// CHECK-LABEL: func.func @parallel_task_taskwait
// CHECK:       call @GOMP_parallel
