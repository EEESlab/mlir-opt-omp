// schedule(dynamic) is NOT implemented for pmsis: the only wsloop construct the
// pmsis rules declare is guarded by `when schedule == static`, so no construct
// matches and the DSL evaluation fails.  Same guard, same outcome, as iomp —
// see schedule-dynamic-unsupported-iomp.mlir.  libgomp is the one runtime that
// has the clause (schedule-dynamic-libgomp.mlir).
//
// The guard is what this test really pins.  pmsis used to declare an unguarded
// `construct wsloop`, so this input matched and was lowered as a static block
// distribution: the loop ran, the answer was not the one a dynamic schedule
// gives, and nothing was reported.  Silent mislowering is the one outcome worse
// than not supporting the clause, hence the hard error below.
//
// The cluster has no dispatch API to build a real dynamic distribution on, so
// implementing one means adding it first (an ext_pi_cl_dynamic_init /
// ext_pi_cl_dynamic_next pair over a shared cursor).  When that lands, replace
// this with a test checking that sequence.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

func.func @wsloop_dynamic() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    // expected-error @below {{wsloop DSL evaluation failed}}
    omp.wsloop schedule(dynamic) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
