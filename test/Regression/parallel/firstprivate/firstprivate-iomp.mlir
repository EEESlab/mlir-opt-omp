// An explicit `firstprivate` clause on a parallel region under iomp.  The
// source variable is injected as a use by OmpToOmpLowerPass so it is collected
// as a capture, the microtask receives it as a pointer parameter, and the
// prolog copies its value into a thread-private alloca before the region body
// runs.  The privatizer block arg is rewritten to that alloca and dropped, so
// it must not survive as a trailing microtask parameter — __kmpc_fork_call
// passes captures, not privatizer slots.
//
// This is the parallel counterpart of task-firstprivate-iomp.mlir, which pins
// the same property on the shareds ABI.
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

func.func @parallel_fp(%arg0: !llvm.ptr) {
  omp.parallel private(@fp_i32 %arg0 -> %p : !llvm.ptr) {
    %v = llvm.load %p : !llvm.ptr -> i32
    llvm.call @use(%v) : (i32) -> ()
    omp.terminator
  }
  return
}

// Exactly three parameters — gtid, btid, and the one capture carrying the
// firstprivate source.  Anchoring on `)` after the third forbids a fourth,
// which is what a leaked privatizer arg would look like.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:    (%{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr {llvm.noalias})
// The copy-in: a private slot, and the source value loaded and stored into it.
// Binding both names is what makes this a real check — it ties the store to
// that alloca rather than just noting that some alloca and some store exist.
// CHECK:         %[[PRIV:.*]] = llvm.alloca
// CHECK:         %[[V:.*]] = llvm.load
// CHECK:         llvm.store %[[V]], %[[PRIV]]
// The region is forked, and the capture is handed over.
// CHECK-LABEL: func.func @parallel_fp
// CHECK:         llvm.call @__kmpc_fork_call
