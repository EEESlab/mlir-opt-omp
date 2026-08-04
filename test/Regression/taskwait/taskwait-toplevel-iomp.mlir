// A top-level omp.taskwait (NOT inside a parallel) reaches the outlining pass
// as an empty-body omp_lower.construct.  It must be lowered with a REAL gtid
// from __kmpc_global_thread_num — not the undef that PlanLoweringPass would emit
// (which crashes the iomp runtime).  This is handled by lowerTopLevelLeaf in the
// outlining pass, so after --omp-outline the construct is already a concrete
// call sequence.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

func.func @tw() {
  omp.taskwait
  return
}

// The taskwait is fully lowered here (no leftover omp_lower.construct for
// PlanLoweringPass), with the gtid coming from __kmpc_global_thread_num:
// CHECK-LABEL: func.func @tw
// CHECK:       call @__kmpc_global_thread_num
// CHECK:       call @__kmpc_omp_taskwait
// CHECK-NOT:   omp_lower.construct
