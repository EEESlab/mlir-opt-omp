// The schedule chunk is the one value in a chunked loop's vocabulary whose type
// the input chooses: omp.wsloop puts no constraint on the operand, so an
// index-typed one is valid MLIR.  It has no conversion into the runtime's index
// type, and it used to reach that conversion and abort on a cast assertion —
// a crash rather than a diagnostic, on input the verifier accepts.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

func.func @wsloop_dynamic_index_chunk() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    %chk = arith.constant 4 : index
    // expected-error @below {{the schedule chunk must be an integer, got 'index'}}
    omp.wsloop schedule(dynamic = %chk : index) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
