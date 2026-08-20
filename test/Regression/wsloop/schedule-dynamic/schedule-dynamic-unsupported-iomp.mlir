// schedule(dynamic) is NOT implemented: rules.dsl declares only
// `construct wsloop when schedule == static`, so no construct matches and the
// DSL evaluation fails.
//
// This test pins the current behaviour, which is the safe one: a hard error, not
// a silent lowering as static.  When dynamic support lands (a
// __kmpc_dispatch_init_4 / __kmpc_dispatch_next_4 construct), this test should
// be replaced by one checking that sequence.
//
// pmsis carries the same guard and gives the same error, for the same reason:
// its `emit thread_bounds` distribution can only be a static one
// (schedule-dynamic-unsupported-pmsis.mlir).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
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
