// firstprivate SNAPSHOT TIMING (libgomp) — KNOWN GAP (XFAIL).
//
// Same property as task-firstprivate-snapshot-iomp.mlir: a firstprivate value
// must be snapshotted into the task's data block at task CREATION.  The libgomp
// task path reuses the packed privatizer handling shared with omp.parallel,
// which captures the source BY POINTER and dereferences it at task ENTRY.  For
// `parallel` that is harmless (no mutation window between fork and region), but
// for a deferred task it observes a post-creation mutation — the wrong value.
//
// Fixing it means forcing scalar firstprivate sources to by-value packing in the
// packed path too (as outlineTaskShareds now does for iomp), without disturbing
// the parallel path.  Until then this is XFAIL: the by-value snapshot load is
// absent at the call site.  Remove the XFAIL line once the packed path snapshots.
//
// XFAIL: *
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
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
  %x  = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr
  %c7 = llvm.mlir.constant(7 : i32) : i32
  llvm.store %c7, %x : i32, !llvm.ptr
  omp.task private(@fp_i32 %x -> %px : !llvm.ptr) {
    %v = llvm.load %px : !llvm.ptr -> i32
    llvm.call @use(%v) : (i32) -> ()
    omp.terminator
  }
  return
}

// The source scalar must be loaded by value at the call site (snapshot) before
// GOMP_task, rather than the source pointer being packed for a deref at entry:
// CHECK-LABEL: func.func @snapshot
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   call @GOMP_task
