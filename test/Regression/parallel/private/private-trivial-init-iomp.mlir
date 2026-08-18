// The `private` recipe shape ClangIR actually emits.
//
// private-iomp.mlir writes the recipe the way a person would, with no `init`
// region.  The front-end always emits one, and for a scalar `private(j)` its
// body is just `omp.yield(%alloc)` — it hands back the storage it was given.
// Same meaning, so it must lower the same way.
//
// Not a hypothetical shape: every PolyBench kernel with a `private` clause
// arrives like this, so refusing it refuses every real input while the
// hand-written tests stay green.  That happened once, which is why it is
// pinned here rather than left to the integration harness.
//
// private-unsupported-init-iomp.mlir is the other side: an init region that
// does something is still refused.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

omp.private {type = private} @priv_i32 : i32 init {
^bb0(%mold: !llvm.ptr, %alloc: !llvm.ptr):
  omp.yield(%alloc : !llvm.ptr)
}

llvm.func @use(i32)

func.func @parallel_priv(%arg0: !llvm.ptr) {
  omp.parallel private(@priv_i32 %arg0 -> %p : !llvm.ptr) {
    %c7 = llvm.mlir.constant(7 : i32) : i32
    llvm.store %c7, %p : i32, !llvm.ptr
    %v = llvm.load %p : !llvm.ptr -> i32
    llvm.call @use(%v) : (i32) -> ()
    omp.terminator
  }
  return
}

// Same outcome as the region-less recipe: gtid and btid only, a slot allocated
// in the body, and no copy-in ahead of the region's own store.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:    (%{{[^:]+}}: !llvm.ptr, %{{[^:)]+}}: !llvm.ptr)
// CHECK:         %[[PRIV:.*]] = llvm.alloca
// CHECK-NOT:     llvm.store {{.*}}, %[[PRIV]]
// CHECK:         %[[C7:.*]] = llvm.mlir.constant(7 : i32)
// CHECK:         llvm.store %[[C7]], %[[PRIV]]
