// pmsis has no lowering for the `if` clause on a parallel region: the DSL
// invoke calls ext_pi_cl_team_fork with a fixed team size and never references
// if_clause, and the GCC-style num_threads select is gated on GOMP_parallel.
// Silently dropping the clause would change semantics (if(false) must run the
// region serialized), so the outlining pass warns instead.
//
// This is the deliberate counterpart to parallel-if-{iomp,libgomp}.mlir: the
// clause is diagnosed here, lowered there.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

llvm.func @use(!llvm.ptr)

func.func @parallel_if(%arg0: !llvm.ptr, %cond: i1) {
  // expected-warning @below {{`if` clause is not supported by this runtime/construct lowering and was ignored}}
  omp.parallel if(%cond) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}
