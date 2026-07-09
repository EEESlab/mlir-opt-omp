// An omp.task with an if clause under iomp branches on the runtime value of
// the condition: when true the task is scheduled deferred via __kmpc_omp_task;
// when false it must run undeferred on the spawning thread, via the
// __kmpc_omp_task_begin_if0 / direct entry call / __kmpc_omp_task_complete_if0
// protocol.  The task is allocated (and shareds populated) either way.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @task_if(%arg0: !llvm.ptr, %cond: i1) {
  omp.task if(%cond) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// The body is outlined into the kmp task-entry signature, returning i32.
// CHECK: func.func {{.*}}@[[ENTRY:outlined_task_[0-9]+]](%{{.*}}: i32, %{{.*}}: !llvm.ptr) -> i32

// The call site allocates the task unconditionally, then branches on the
// condition value.
// CHECK-LABEL: func.func @task_if
// CHECK:       %[[GTID:.*]] = call @__kmpc_global_thread_num
// CHECK:       %[[TASK:.*]] = call @__kmpc_omp_task_alloc
// CHECK:       llvm.cond_br %{{.*}}, ^[[DEF:bb[0-9]+]], ^[[IF0:bb[0-9]+]]

// true → deferred scheduling.
// CHECK:       ^[[DEF]]:
// CHECK:         call @__kmpc_omp_task({{.*}}%[[GTID]], %[[TASK]])
// CHECK:         llvm.br ^[[CONT:bb[0-9]+]]

// false → undeferred: begin_if0, direct call of the entry, complete_if0.
// CHECK:       ^[[IF0]]:
// CHECK:         call @__kmpc_omp_task_begin_if0({{.*}}%[[GTID]], %[[TASK]])
// CHECK:         call @[[ENTRY]](%[[GTID]], %[[TASK]])
// CHECK:         call @__kmpc_omp_task_complete_if0({{.*}}%[[GTID]], %[[TASK]])
// CHECK:         llvm.br ^[[CONT]]
