// `firstprivate` on a parallel region under pmsis.  Same packed capture
// strategy as libgomp — the source rides in the capture struct and is copied
// into a per-core private slot in the prolog — but reached through a different
// DSL invoke (ext_pi_cl_team_fork), so the call site is worth pinning
// separately from firstprivate-libgomp.mlir.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
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

// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:    (%{{[^:)]+}}: !llvm.ptr)
// CHECK:         %[[PRIV:.*]] = llvm.alloca
// CHECK:         %[[V:.*]] = llvm.load
// CHECK:         llvm.store %[[V]], %[[PRIV]]
// CHECK-LABEL: func.func @parallel_fp
// CHECK:         call @ext_pi_cl_team_fork
