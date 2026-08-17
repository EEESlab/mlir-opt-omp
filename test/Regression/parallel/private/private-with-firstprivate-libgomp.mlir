// The mixed `private` + `firstprivate` case of
// private-with-firstprivate-iomp.mlir, on the packed ABI.  The capture struct
// is what makes the property visible here: it must hold exactly one field, the
// firstprivate source.  A private that still carried a block arg into the
// outlining pass would either add a field or make the copy-in read the wrong
// one, since the packed loop pairs privatizer arg i with loaded capture i.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
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

// The closure keeps its single data pointer...
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:    (%[[DATA:[^:)]+]]: !llvm.ptr)
// ...and the struct behind it has exactly one field — a one-element `(ptr)`,
// not `(ptr, ptr)`.  That single field is the firstprivate source, unpacked
// and copied into the firstprivate slot.
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[DATA]][0, 0] : {{.*}} !llvm.struct<(ptr)>
// CHECK:         %[[SRC:.*]] = llvm.load %[[GEP]]
// CHECK:         %[[FP:.*]] = llvm.alloca
// CHECK:         %[[V:.*]] = llvm.load %[[SRC]]
// CHECK:         llvm.store %[[V]], %[[FP]]
// The private slot, with no copy-in ahead of the region's own store.
// CHECK:         %[[PRIV:.*]] = llvm.alloca
// CHECK-NOT:     llvm.store {{.*}}, %[[PRIV]]
// CHECK:         %[[C7:.*]] = llvm.mlir.constant(7 : i32)
// CHECK:         llvm.store %[[C7]], %[[PRIV]]

// CHECK-LABEL: func.func @mixed
// CHECK:         call @GOMP_parallel
