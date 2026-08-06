// num_threads on pmsis is NOT honoured: rules.dsl calls
// `ext_pi_cl_team_fork(8, body, env_ptr)` with the team size written as a
// literal, so the clause value never reaches the fork and the region always
// runs on 8 cores.  Nothing diagnoses it — the clause is accepted and dropped.
//
// The check below states the behaviour the lowering should have: the fork takes
// the value the user asked for.  It fails today, hence the XFAIL; when the DSL
// starts threading num_threads through (a `when has(num_threads)` select like
// the libgomp parallel does with GOMP_parallel), this XPASSes and the XFAIL
// line goes away.
//
// XFAIL: *
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @parallel_nt(%arg0: !llvm.ptr, %nt: i32) {
  omp.parallel num_threads(%nt : i32) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// Capture the clause operand off the signature, then require the fork to be
// handed that same value.  Today the first argument is a constant 8, so the
// substitution does not match.
// CHECK-LABEL: func.func @parallel_nt
// CHECK-SAME:    %{{[^:]+}}: !llvm.ptr, %[[NT:[^:]+]]: i32
// CHECK:         call @ext_pi_cl_team_fork(%[[NT]],
