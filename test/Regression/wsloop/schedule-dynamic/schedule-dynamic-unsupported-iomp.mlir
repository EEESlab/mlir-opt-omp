// schedule(dynamic) is NOT implemented for iomp: the only wsloop construct the
// iomp rules declare is guarded by `when schedule == static`, so no construct
// matches and the DSL evaluation fails.
//
// This test pins the current behaviour, which is the safe one: a hard error, not
// a silent lowering as static.  When dynamic support lands (a
// __kmpc_dispatch_init_4 / __kmpc_dispatch_next_4 construct), this test should
// be replaced by one checking that sequence.
//
// libgomp already has it — schedule-dynamic-libgomp.mlir — so what is missing
// here is one construct in rules.dsl, not a lowering: the chunked loop the pass
// builds around it is the same one, and iomp needs neither a first_chunk block
// (its dispatch_next opens the loop as well as repeating it) nor any of the
// chunk_* properties (its ABI is what they default to).
//
// pmsis is guarded too and errors the same way, for a different reason: its
// `emit thread_bounds` distribution can only ever be a static one
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
