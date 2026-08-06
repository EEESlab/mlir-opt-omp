// A pure `private` clause on pmsis.  pmsis shares the packed capture strategy
// with libgomp, so it hits the same gap described in private-libgomp.mlir: no
// capture is produced for a private, the copy-in loop is skipped, and the
// surviving privatizer block arg is reported as an unsupported clause shape.
//
// It gets its own test rather than being folded into the libgomp one because
// the two runtimes reach the packed path through different DSL constructs
// (ext_pi_cl_team_fork vs GOMP_parallel), and a fix that threads the private
// through one call site says nothing about the other.
//
// XFAIL: *
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
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

// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:    (%{{[^:)]+}}: !llvm.ptr)
// The private slot, and no copy-in written into it before the region's store.
// CHECK:         %[[PRIV:.*]] = llvm.alloca
// CHECK-NOT:     llvm.store {{.*}}, %[[PRIV]]
// CHECK:         %[[C7:.*]] = llvm.mlir.constant(7 : i32)
// CHECK:         llvm.store %[[C7]], %[[PRIV]]
// CHECK-LABEL: func.func @parallel_priv
// CHECK:         call @ext_pi_cl_team_fork
