// A top-level omp.taskwait (NOT inside a parallel) has no body and no captures,
// so the outlining pass has nothing to do with it: it survives as an empty-body
// omp_lower.construct and PlanLoweringPass turns it into calls.  This is the
// only construct shape that exercises the full three-pass pipeline, since
// everything with a region is consumed by the outlining pass.
//
// It must get a REAL gtid from __kmpc_global_thread_num — an undef there
// crashes the iomp runtime.  That used to force this lowering into the outlining
// pass; now PlanLoweringPass resolves ident/%gtid through the same shared
// vocabulary (see lib/Transforms/PlanEmit.h).
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

func.func @tw() {
  omp.taskwait
  return
}

// CHECK-LABEL: func.func @tw
// CHECK:       call @__kmpc_global_thread_num
// CHECK:       call @__kmpc_omp_taskwait
// CHECK-NOT:   omp_lower.construct
