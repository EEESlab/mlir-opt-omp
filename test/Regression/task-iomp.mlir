// An omp.task under the iomp runtime lowers to the __kmpc_omp_task_alloc /
// __kmpc_omp_task sequence: the body is outlined into an i32(i32 gtid, ptr task)
// entry that reads its captures from task->shareds, and the call site allocates
// the task, populates shareds, and schedules it.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s
//
// The DSL-owned kmp_task_t layout reaches the construct's prop_dict as the
// symbolic `%struct:...` token, which the outlining pass re-expands.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower | FileCheck %s --check-prefix=LOWER
// LOWER: omp_lower.construct
// LOWER-SAME: kmp_task_t = "%struct:ptr,ptr,i32,ptr,ptr"

llvm.func @use(!llvm.ptr)

func.func @task(%arg0: !llvm.ptr) {
  omp.task {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// The body is outlined into the kmp task-entry signature, returning i32.
// CHECK: func.func {{.*}}@outlined_task_{{[0-9]+}}(%{{.*}}: i32, %{{.*}}: !llvm.ptr) -> i32
// CHECK:   llvm.call @use

// The call site allocates the task and schedules it.
// CHECK-LABEL: func.func @task
// CHECK:       call @__kmpc_global_thread_num
// CHECK:       call @__kmpc_omp_task_alloc
// CHECK:       call @__kmpc_omp_task
