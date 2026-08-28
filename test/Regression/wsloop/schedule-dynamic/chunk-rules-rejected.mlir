// The four ways a chunked wsloop construct can be written wrong in the rules,
// and the diagnostic each one gets.  All four used to be silent: the first
// dropped the chunk blocks and ran the whole iteration space on every thread,
// the second built a loop with nothing to turn on, and the last two fell back
// to a default — for chunk_bound, one that runs an iteration past the end of
// every chunk.  A rule file is not user input, but it is edited by hand and
// read at run time with no build to catch a typo, so the pass has to be what
// catches it.
//
// The rule file is Inputs/chunk-malformed.dsl, not the shipped rules.dsl; the
// schedule kind on each loop below selects one of its four broken constructs.
// One run reaches all four, which is also what pins that the pass keeps going
// after a bad loop instead of reporting them one run at a time.
//
// RUN: not mlir-opt-omp %s --omp-lower-dsl=%S/../../Inputs/chunk-malformed.dsl \
// RUN:   --omp-lower-runtime=libgomp --omp-to-omp-lower --omp-outline \
// RUN:   --verify-diagnostics

func.func @first_chunk_without_next() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    // expected-error @below {{declares `first_chunk` but no `next_chunk`}}
    omp.wsloop schedule(dynamic) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

func.func @next_chunk_without_a_call() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    // expected-error @below {{chunk block of this wsloop makes no call}}
    omp.wsloop schedule(guided) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

func.func @misspelled_chunk_bound() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    // expected-error @below {{`chunk_bound = exclusve` is neither `inclusive` nor `exclusive`}}
    omp.wsloop schedule(auto) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

func.func @unknown_chunk_index() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    // expected-error @below {{`chunk_index = i62` is not a type this lowering knows}}
    omp.wsloop schedule(runtime) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// `not` on the RUN line is half the point: each diagnostic has to come with a
// non-zero exit, or the pipeline carries on with a module that still holds an
// unlowered omp.wsloop.
