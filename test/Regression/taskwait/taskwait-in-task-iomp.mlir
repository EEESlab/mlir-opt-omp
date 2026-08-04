// A taskwait *directly inside a task body* (here `parallel { task { taskwait } }`)
// rides into the task's outlined entry as an omp.taskwait and is lowered there.
// The iomp task entry uses the shareds ABI i32(i32 gtid, ptr task), so %gtid is
// arg 0 by value; the taskwait becomes __kmpc_omp_taskwait(ident, gtid).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @task_with_taskwait(%arg0: !llvm.ptr) {
  omp.parallel {
    omp.task {
      llvm.call @use(%arg0) : (!llvm.ptr) -> ()
      omp.taskwait
      omp.terminator
    }
    omp.terminator
  }
  return
}

// The taskwait is lowered inside the task entry, using the entry's gtid arg.
// CHECK: func.func {{.*}}@outlined_task_{{[0-9]+}}(%[[GTID:.*]]: i32, %{{.*}}: !llvm.ptr) -> i32
// CHECK:   llvm.call @use
// CHECK:   call @__kmpc_omp_taskwait
