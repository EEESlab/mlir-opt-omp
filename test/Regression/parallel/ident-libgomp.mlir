// ident_t globals are an iomp-only concept (__kmpc_* take a loc argument).
// The libgomp path (GOMP_* API) must NOT emit any ident_t or psource global,
// even when run through the full outlining pipeline.
// See docs/lowering-specs/ident-lowering-spec.md.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s
//
// A second pass with its own prefix anchors the absence checks above: on their
// own they would also hold for an empty module, so they cannot tell "no ident
// because libgomp needs none" from "no ident because nothing was lowered".
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan \
// RUN:   | FileCheck %s --check-prefix=LOWERED

func.func @parallel_with_barrier() {
  omp.parallel {
    omp.barrier
    omp.terminator
  }
  return
}

// CHECK-NOT: @__omp_ident_
// CHECK-NOT: @__omp_src_loc_default

// The region really was lowered — the fork is there.  (The barrier is the
// region's trailing op, so --omp-barrier-elim would drop it as redundant with
// the implicit join; this run does not enable the pass, and GOMP_barrier
// carries no ident either way.)
// LOWERED: call @GOMP_parallel
