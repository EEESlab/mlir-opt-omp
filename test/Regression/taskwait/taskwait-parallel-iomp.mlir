// An omp.taskwait inside a parallel rides into the parallel's outlined function
// and is lowered there (the same in-outlined-fn path as omp.barrier) to
// __kmpc_omp_taskwait(ident, gtid).  Here it follows a spawned task, the common
// `parallel { task; taskwait }` shape.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

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
// parallel's outlined function.
// CHECK: func.func {{.*}}@outlined_parallel_{{[0-9]+}}
// CHECK:   call @__kmpc_omp_task
// CHECK:   call @__kmpc_omp_taskwait

// The fork itself lands in the original function.
// CHECK-LABEL: func.func @parallel_task_taskwait
// CHECK:       @__kmpc_fork_call
