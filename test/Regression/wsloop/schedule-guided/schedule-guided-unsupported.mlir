// schedule(guided) is not implemented anywhere: no runtime declares a wsloop
// construct whose guard matches it, so the evaluation finds nothing and fails.
// Same for auto and runtime, which no rule mentions either.
//
// What this pins is the guard, not the missing feature.  Every wsloop construct
// in rules.dsl carries a `when schedule == ...`, and if one ever loses it the
// unmatched kinds start matching it instead — quietly lowered as whatever that
// construct happens to be.  pmsis shipped exactly that bug once; guarding its
// construct is what turned it into an error (schedule-dynamic-unsupported-
// pmsis.mlir).  With iomp and libgomp now having two constructs each, this is
// the test that says the second one did not widen the net.
//
// On iomp it would take a __kmpc_dispatch_init_4 with 36 in place of 35, and on
// libgomp the GOMP_loop_guided_* pair; the chunked loop the pass builds around
// either is the one already there.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

func.func @wsloop_guided() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    // expected-error @below {{wsloop DSL evaluation failed}}
    omp.wsloop schedule(guided) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
