// firstprivate SNAPSHOT TIMING (iomp).
//
// A firstprivate value must be snapshotted into the task's data block at task
// CREATION, not read through a captured pointer at task ENTRY.  Otherwise a
// deferred task that runs after the source was mutated (the canonical
// `firstprivate(i)` spawn-loop idiom) would observe the wrong value.
//
// This is a runtime-scheduling-sensitive bug, so it cannot be pinned by an
// end-to-end run (an eager task would read the right value even when broken).
// Instead we assert the structural fix: the source scalar is captured BY VALUE,
// so at the call site — right after __kmpc_omp_task_alloc, while populating
// shareds — it is loaded (snapshotted) and stored into the shareds block.  The
// broken lowering stores the source POINTER instead, with no i32 load here.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

omp.private {type = firstprivate} @fp_i32 : i32 copy {
^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
  %v = llvm.load %src : !llvm.ptr -> i32
  llvm.store %v, %dst : i32, !llvm.ptr
  omp.yield(%dst : !llvm.ptr)
}

llvm.func @use(i32)

func.func @snapshot() {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %x  = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr    // firstprivate source (local)
  %c7 = llvm.mlir.constant(7 : i32) : i32
  llvm.store %c7, %x : i32, !llvm.ptr
  omp.task private(@fp_i32 %x -> %px : !llvm.ptr) {
    %v = llvm.load %px : !llvm.ptr -> i32
    llvm.call @use(%v) : (i32) -> ()
    omp.terminator
  }
  return
}

// The entry keeps its two-parameter ABI:
// CHECK: func.func {{.*}}@outlined_task_{{[0-9]+}}(%{{[^,)]*}}: i32, %{{[^,)]*}}: !llvm.ptr) -> i32

// At the call site the firstprivate value is snapshotted: after allocating the
// task, the source scalar is loaded by value (not the pointer) and stored into
// shareds, before scheduling the task.
// CHECK-LABEL: func.func @snapshot
// CHECK:   call @__kmpc_omp_task_alloc
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   call @__kmpc_omp_task
