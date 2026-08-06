// The libgomp runtime block in rules.dsl never mentions proc_bind, so the
// clause is accepted and dropped with no call and no diagnostic.  That is a
// silent semantic change: proc_bind(close) asks for a thread-affinity policy
// and the generated code applies none.
//
// Dropping it may well be the right end state for libgomp — GOMP has no
// per-region push API, affinity comes from OMP_PROC_BIND — but then it must be
// *said*, not implied by silence.  This test asks for the same treatment the
// `if` clause already gets on pmsis (parallel-if-pmsis.mlir): a warning that
// the clause was ignored.  It fails today, hence the XFAIL.
//
// XFAIL: *
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

llvm.func @use(!llvm.ptr)

func.func @parallel_pb(%arg0: !llvm.ptr) {
  // expected-warning @below {{`proc_bind` clause is not supported by this runtime/construct lowering and was ignored}}
  omp.parallel proc_bind(close) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}
