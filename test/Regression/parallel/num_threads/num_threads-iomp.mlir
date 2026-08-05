// num_threads on iomp is a `pre` call: __kmpc_push_num_threads sets the team
// size for the *next* fork, so it must be emitted before __kmpc_fork_call.
// The pre block is guarded by `when has(num_threads)` in rules.dsl, so a
// parallel without the clause must emit no push at all (and, since the push is
// the only pre call here, no global_tid either — it is materialised on demand).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @parallel_nt(%arg0: !llvm.ptr, %nt: i32) {
  omp.parallel num_threads(%nt : i32) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

func.func @parallel_plain(%arg0: !llvm.ptr) {
  omp.parallel {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// With the clause: push before the fork, and the gtid it needs is materialised.
// CHECK-LABEL: func.func @parallel_nt
// CHECK:         call @__kmpc_global_thread_num
// CHECK:         call @__kmpc_push_num_threads
// CHECK:         llvm.call @__kmpc_fork_call

// Without it: no push, and no gtid call — nothing else in this path needs one.
// CHECK-LABEL: func.func @parallel_plain
// CHECK-NOT:     __kmpc_push_num_threads
// CHECK-NOT:     __kmpc_global_thread_num
// CHECK:         llvm.call @__kmpc_fork_call
