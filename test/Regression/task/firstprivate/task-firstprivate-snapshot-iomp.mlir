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
// so at the call site it is loaded (snapshotted) and that scalar — not the
// source POINTER — is what gets stored into the shareds block.
//
// The two halves come from different passes, which is why the load precedes the
// allocation: the outlining pass resolves what each capture field receives (it
// alone knows the classification) and the plan pass writes the values into the
// block the runtime just handed back.  Only their order changes; the snapshot
// still happens at task creation, which is what the test is about.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

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

// At the call site the firstprivate value is snapshotted: the source scalar is
// loaded by value (not the pointer), and that same value is the one stored into
// the shareds block, before the task is scheduled.
// CHECK-LABEL: func.func @snapshot
// CHECK:   %[[SNAP:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   call @__kmpc_omp_task_alloc
// CHECK:   llvm.store %[[SNAP]], %{{.*}} : i32, !llvm.ptr
// CHECK:   call @__kmpc_omp_task
