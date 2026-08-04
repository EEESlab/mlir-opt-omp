// KNOWN GAP — this test is expected to fail until pmsis stops accepting
// schedule(dynamic).
//
// iomp and libgomp guard their loop construct with `when schedule == static`,
// so schedule(dynamic) finds no matching construct and is rejected outright
// (schedule-dynamic-unsupported-iomp.mlir pins that).  pmsis declares a plain
// `construct wsloop` with no guard, so the same input matches and is lowered as
// a static block distribution: the loop runs, the results are wrong-ish (a
// dynamic schedule is not a static one) and nothing is reported.
//
// Silent mislowering is the one outcome worse than not supporting the clause,
// so the assertion below is what the tool SHOULD do.  Two ways to make this
// test go green, both acceptable:
//   - guard the pmsis construct with `when schedule == static` too, so the
//     evaluation fails like the other runtimes, or
//   - implement a real dynamic distribution for the cluster.
// Either way, delete the XFAIL line when it starts passing.
//
// XFAIL: *
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

func.func @wsloop_dynamic() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    // expected-error @below {{schedule(dynamic) is not supported}}
    omp.wsloop schedule(dynamic) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
