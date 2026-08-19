// A bare omp.barrier lowers to an omp_lower.construct carrying the iomp
// __kmpc_barrier call.
//
// Two runs: the first reads the call name out of the plan attribute (what the
// rules decided), the second takes the pipeline down to emission (what came
// out).  parallel/ident-iomp.mlir covers a barrier *inside* a parallel, where
// the gtid comes from the microtask signature; a top-level one has no enclosing
// outlined function, so ident and gtid must be materialised from scratch —
// a different path, and an undef gtid here would crash the runtime.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower | FileCheck %s
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan \
// RUN:   | FileCheck %s --check-prefix=LOWERED

func.func @barrier_only() {
  omp.barrier
  return
}

// The plan the rules produced:
// CHECK-LABEL: func.func @barrier_only
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "iomp"
// CHECK-SAME:    construct = "barrier"
// CHECK-SAME:    __kmpc_barrier

// What actually came out: a real thread id, then the barrier taking it.
// LOWERED-LABEL: func.func @barrier_only
// LOWERED:         call @__kmpc_global_thread_num
// LOWERED:         call @__kmpc_barrier
// LOWERED-NOT:     omp_lower.construct
