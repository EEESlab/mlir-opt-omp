// The same input under the libgomp runtime selects GOMP_barrier instead.
//
// Two runs, because they check different things.  The first stops at
// --omp-to-omp-lower and reads GOMP_barrier out of the *plan attribute*: it says
// the rules picked the right call.  That is not the same as the call being
// emitted, so the second run takes the pipeline all the way down and checks the
// real one — the failure mode test/README.md warns about, and the only place
// this construct's emission is covered.  (The GOMP_barrier in nowait-libgomp.mlir
// comes from the wsloop `post` block, not from a barrier construct.)
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower | FileCheck %s
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan \
// RUN:   | FileCheck %s --check-prefix=LOWERED

func.func @barrier_only() {
  omp.barrier
  return
}

// The plan the rules produced:
// CHECK-LABEL: func.func @barrier_only
// CHECK:       omp_lower.construct
// CHECK-SAME:    runtime = "libgomp"
// CHECK-SAME:    construct = "barrier"
// CHECK-SAME:    GOMP_barrier

// What actually came out.  GOMP_barrier takes no arguments — no ident, no gtid,
// unlike the iomp spelling — and the construct must be gone once lowered.
// LOWERED-LABEL: func.func @barrier_only
// LOWERED:         call @GOMP_barrier() : () -> ()
// LOWERED-NOT:     omp_lower.construct
