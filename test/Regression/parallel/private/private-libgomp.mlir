// The same pure `private` clause as private-iomp.mlir, on the packed ABI.
// Expected: the closure takes only its data pointer and allocates the private
// slot itself, with no copy-in.
//
// Today it does not get that far.  The packed copy-in loop sits inside
// `if (!captures.empty())`, and a pure private produces no capture (only
// firstprivate sources are injected as uses), so the loop never runs, the
// privatizer block arg keeps its uses, and OmpOutliningPass reports
// "unsupported private/firstprivate clause; outlined ABI would break".
//
// So the failure mode differs from iomp: a hard error here, a silent read of an
// unfilled argument there.  Both are wrong for a clause the support matrix
// lists as working, and the fix is shared — teach the privatizer handling to
// tell `private` from `firstprivate` instead of treating every privatizer arg
// as a copy-in.
//
// XFAIL: *
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
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

// The closure keeps its single data-pointer parameter — a private must not
// leak into the signature as a trailing arg GOMP_parallel would never fill.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:    (%{{[^:)]+}}: !llvm.ptr)
// The private slot is allocated in the outlined body and reaches the region's
// own store with nothing written into it in between — a copy-in would land in
// that gap, which is what separates private from firstprivate here.
// CHECK:         %[[PRIV:.*]] = llvm.alloca
// CHECK-NOT:     llvm.store {{.*}}, %[[PRIV]]
// CHECK:         %[[C7:.*]] = llvm.mlir.constant(7 : i32)
// CHECK:         llvm.store %[[C7]], %[[PRIV]]
// The region really was lowered.
// CHECK-LABEL: func.func @parallel_priv
// CHECK:         call @GOMP_parallel
