// ident_t globals are an iomp-only concept (__kmpc_* take a loc argument).
// The libgomp path (GOMP_* API) must NOT emit any ident_t or psource global,
// even when run through the full outlining pipeline.
// See docs/lowering-specs/ident-lowering-spec.md.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

func.func @parallel_with_barrier() {
  omp.parallel {
    omp.barrier
    omp.terminator
  }
  return
}

// CHECK-NOT: @__omp_ident_
// CHECK-NOT: @__omp_src_loc_default
