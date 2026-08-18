// A `private` and a `firstprivate` on the same parallel, with the private
// listed *first*.  The order is the point: only firstprivate sources become
// captures, and OmpOutliningPass pairs privatizer block arg i with capture i
// positionally, so a private still holding a block arg would shift that
// pairing and make the firstprivate copy in from the wrong slot.  It cannot,
// because wirePrivatizers allocates the private and drops its block arg before
// the construct is built, leaving the two lists aligned by construction.
//
// private-with-firstprivate-libgomp.mlir pins the same on the packed ABI.  No
// pmsis variant: it shares that path, and what differs is the call site, which
// private-pmsis.mlir already covers.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

omp.private {type = private} @priv_i32 : i32

omp.private {type = firstprivate} @fp_i32 : i32 copy {
^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
  %v = llvm.load %src : !llvm.ptr -> i32
  llvm.store %v, %dst : i32, !llvm.ptr
  omp.yield(%dst : !llvm.ptr)
}

llvm.func @use(i32)

func.func @mixed(%a: !llvm.ptr, %b: !llvm.ptr) {
  omp.parallel private(@priv_i32 %a -> %p, @fp_i32 %b -> %f : !llvm.ptr, !llvm.ptr) {
    %c7 = llvm.mlir.constant(7 : i32) : i32
    llvm.store %c7, %p : i32, !llvm.ptr
    %pv = llvm.load %p : !llvm.ptr -> i32
    %fv = llvm.load %f : !llvm.ptr -> i32
    llvm.call @use(%pv) : (i32) -> ()
    llvm.call @use(%fv) : (i32) -> ()
    omp.terminator
  }
  return
}

// Exactly three parameters — gtid, btid, and the single capture carrying the
// firstprivate source.  Anchoring on `)` forbids a fourth: the private must
// contribute neither a capture nor a leaked privatizer arg.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:    (%{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %[[SRC:[^:]+]]: !llvm.ptr {llvm.noalias})
// The firstprivate is copied in from that capture, not from some other slot.
// CHECK:         %[[FP:.*]] = llvm.alloca
// CHECK:         %[[V:.*]] = llvm.load %[[SRC]]
// CHECK:         llvm.store %[[V]], %[[FP]]
// The private gets its own slot with nothing written into it before the
// region's own store.
// CHECK:         %[[PRIV:.*]] = llvm.alloca
// CHECK-NOT:     llvm.store {{.*}}, %[[PRIV]]
// CHECK:         %[[C7:.*]] = llvm.mlir.constant(7 : i32)
// CHECK:         llvm.store %[[C7]], %[[PRIV]]

// One vararg is forked: the firstprivate source, and nothing for the private.
// CHECK-LABEL: func.func @mixed
// CHECK:         llvm.call @__kmpc_fork_call(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,)]+}}) vararg
