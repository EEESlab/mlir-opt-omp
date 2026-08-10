// pmsis has no push call: num_threads is the team-size argument of
// ext_pi_cl_team_fork, so the clause selects between the `when has(num_threads)`
// and `otherwise` arms of the DSL invoke, exactly as it does for GOMP_parallel.
// Without the clause the slot is `default_team_size`, the 8-core cluster.
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

func.func @parallel_plain(%arg0: !llvm.ptr) {
  omp.parallel {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// With the clause: capture the operand off the signature and require the fork
// to be handed that same value, rather than a team size of its own choosing.
// CHECK-LABEL: func.func @parallel_nt
// CHECK-SAME:    %{{[^:]+}}: !llvm.ptr, %[[NT:[^:]+]]: i32
// CHECK:         call @ext_pi_cl_team_fork(%[[NT]],

// Without it: the default team size goes in instead.
// CHECK-LABEL: func.func @parallel_plain
// CHECK-DAG:     %[[EIGHT:.*]] = arith.constant 8 : i32
// CHECK:         call @ext_pi_cl_team_fork(%[[EIGHT]],
