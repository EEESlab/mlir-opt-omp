// An omp.task with an if clause under libgomp.  Unlike iomp (task-if-iomp.mlir),
// there is no branch at the call site: GOMP_task takes the condition as its
// `_Bool if_clause` parameter and decides between deferred scheduling and
// inline execution itself.  The i1 clause value is zero-extended to i8 to match
// that C parameter.  Without the clause the DSL's `otherwise` arm passes a
// constant true (always deferred).
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

func.func @task_noif(%arg0: !llvm.ptr) {
  omp.task {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// With if(%cond): the i1 is widened to i8 and handed straight to GOMP_task.
// The CHECK-NOT covers the window where iomp would have put its cond_br.
// CHECK-LABEL: func.func @task_if
// CHECK:       %[[IF:.*]] = llvm.zext %{{.*}} : i1 to i8
// CHECK-NOT:   llvm.cond_br
// CHECK:       call @GOMP_task({{.*}}%[[IF]]{{.*}})

// Without the clause: the `otherwise` arm passes a constant true instead.
// CHECK-LABEL: func.func @task_noif
// CHECK:       %[[TRUE:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK:       call @GOMP_task({{.*}}%[[TRUE]]{{.*}})
