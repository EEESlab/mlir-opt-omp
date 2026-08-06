// schedule(dynamic) is NOT implemented for libgomp: rules.dsl declares only
// `construct wsloop when schedule == static`, so no construct matches and the
// DSL evaluation fails.  Same guard, same outcome, as iomp — see
// schedule-dynamic-unsupported-iomp.mlir.
//
// This pins the safe behaviour: a hard error rather than a silent lowering as
// static.  When a dynamic path lands (GOMP_loop_dynamic_start /
// GOMP_loop_dynamic_next), replace this with a test checking that sequence.
//
// The third runtime behaves differently and deliberately so: pmsis declares an
// UNGUARDED `construct wsloop`, so there the same input matches and is
// mislowered as a static block distribution with no diagnostic
// (schedule-dynamic-pmsis.mlir, XFAILed).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
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
