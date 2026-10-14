// Same unsupported firstprivate shape as task-firstprivate-unsupported-libgomp,
// under iomp: the privatizer block arg is used without a scalar load, so the
// copy-in in outlineTaskShareds can't infer the element type and skips it.  The
// surviving block arg would break the i32(i32 gtid, ptr task) entry ABI, so the
// outlining pass diagnoses it instead of miscompiling.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

omp.private {type = firstprivate} @fp_i32 : i32 copy {
^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
  %v = llvm.load %src : !llvm.ptr -> i32
  llvm.store %v, %dst : i32, !llvm.ptr
  omp.yield(%dst : !llvm.ptr)
}

llvm.func @sink(!llvm.ptr)

func.func @unsupported_fp() {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %x  = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr
  // expected-error @+1 {{unsupported private/firstprivate clause}}
  omp.task private(@fp_i32 %x -> %px : !llvm.ptr) {
    llvm.call @sink(%px) : (!llvm.ptr) -> ()   // uses %px by address, never loads it
    omp.terminator
  }
  return
}
