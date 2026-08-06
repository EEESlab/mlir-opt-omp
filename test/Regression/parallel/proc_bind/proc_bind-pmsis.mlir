// Same gap as proc_bind-libgomp.mlir, on pmsis: the runtime block declares no
// proc_bind handling, so the clause is accepted and dropped without a word.
//
// On pmsis the answer is almost certainly "this target has no affinity policy
// to set" — the fork goes to a fixed cluster of cores — but a clause that
// changes nothing must still be reported rather than absorbed, on the same
// grounds as the `if` clause here (parallel-if-pmsis.mlir).  Fails today.
//
// XFAIL: *
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
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
