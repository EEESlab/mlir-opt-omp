// A pure `private` clause on a parallel region: every thread gets its own
// uninitialised copy of the variable.  Unlike firstprivate there is nothing to
// copy in, so the outlined microtask needs no capture for it at all — it should
// allocate the slot itself and take only the ABI-mandated (gtid, btid).
//
// The slot is allocated by OmpToOmpLowerPass (wirePrivatizers), inside the
// region that is about to become the outlined body, and the privatizer block
// arg is dropped there.  That is what keeps it out of the signature: by the
// time OmpOutliningPass builds the microtask there is nothing left to pair
// with a capture.
//
// This test used to XFAIL.  A pure private reached the outlining pass with a
// block arg and no matching capture, and the iomp copy-in loop paired
// privatizerArgs[i] with entry argument `privCapStart + i` — with no captures
// that index landed back on the privatizer arg itself, so the prolog loaded
// from an argument the call site never fills and stored the result into the
// private slot, seeding it with garbage and saying nothing.  The CHECK-NOT
// below is what rules that out.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

omp.private {type = private} @priv_i32 : i32

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

// The microtask takes exactly gtid and btid: a private needs no capture, so no
// third parameter may appear.  Anchoring on `)` after the second arg is what
// forces "exactly two".
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:    (%{{[^:]+}}: !llvm.ptr, %{{[^:)]+}}: !llvm.ptr)
// The private slot is allocated inside the outlined body...
// CHECK:         %[[PRIV:.*]] = llvm.alloca
// ...and nothing writes to it before the region's own store: a copy-in would
// land exactly here, so this CHECK-NOT is what distinguishes private from
// firstprivate.  Binding %[[PRIV]] keeps it about *this* slot.
// CHECK-NOT:     llvm.store {{.*}}, %[[PRIV]]
// CHECK:         %[[C7:.*]] = llvm.mlir.constant(7 : i32)
// CHECK:         llvm.store %[[C7]], %[[PRIV]]
