// `firstprivate` on a parallel region under libgomp: the packed ABI.  The
// source variable travels inside the capture struct, the closure unpacks it
// from the data pointer, and the prolog copies its value into a thread-private
// alloca.  As on iomp the privatizer block arg is rewritten to that alloca and
// erased — here a survivor would add a second closure parameter, which
// GOMP_parallel (body, data) has no way to fill.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

omp.private {type = firstprivate} @fp_i32 : i32 copy {
^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
  %v = llvm.load %src : !llvm.ptr -> i32
  llvm.store %v, %dst : i32, !llvm.ptr
  omp.yield(%dst : !llvm.ptr)
}

llvm.func @use(i32)

func.func @parallel_fp(%arg0: !llvm.ptr) {
  omp.parallel private(@fp_i32 %arg0 -> %p : !llvm.ptr) {
    %v = llvm.load %p : !llvm.ptr -> i32
    llvm.call @use(%v) : (i32) -> ()
    omp.terminator
  }
  return
}

// One parameter only: the closure data pointer.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:    (%{{[^:)]+}}: !llvm.ptr)
// The copy-in, with the store tied to the private slot it writes.
// CHECK:         %[[PRIV:.*]] = llvm.alloca
// CHECK:         %[[V:.*]] = llvm.load
// CHECK:         llvm.store %[[V]], %[[PRIV]]
// The region really was lowered — without this the checks above would also
// hold for a module where nothing happened.
// CHECK-LABEL: func.func @parallel_fp
// CHECK:         call @GOMP_parallel
