// An omp.parallel with an if clause under iomp branches on the runtime value
// of the condition: when true it forks as usual via __kmpc_fork_call; when
// false the region runs serialized on the calling thread, bracketed by
// __kmpc_serialized_parallel / __kmpc_end_serialized_parallel, with the
// microtask called directly (gtid/btid passed by pointer, btid = 0).
// __kmpc_fork_call_if is NOT used: it takes a single packed void* argument
// (argc <= 1), incompatible with the by_pointer capture convention.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @parallel_if(%arg0: !llvm.ptr, %cond: i1) {
  omp.parallel if(%cond) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// CHECK: func.func {{.*}}@[[MICRO:outlined_parallel_[0-9]+]](%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr)

// CHECK-LABEL: func.func @parallel_if
// CHECK:       %[[GTID:.*]] = call @__kmpc_global_thread_num
// CHECK:       llvm.cond_br %{{.*}}, ^[[FORK:bb[0-9]+]], ^[[SER:bb[0-9]+]]

// true → the usual fork.
// CHECK:       ^[[FORK]]:
// CHECK:         llvm.call @__kmpc_fork_call
// CHECK:         llvm.br ^[[CONT:bb[0-9]+]]

// false → serialized parallel with a direct microtask call.
// CHECK:       ^[[SER]]:
// CHECK:         call @__kmpc_serialized_parallel({{.*}}, %[[GTID]])
// CHECK:         call @[[MICRO]](
// CHECK:         call @__kmpc_end_serialized_parallel({{.*}}, %[[GTID]])
// CHECK:         llvm.br ^[[CONT]]

// CHECK-NOT:   __kmpc_fork_call_if
