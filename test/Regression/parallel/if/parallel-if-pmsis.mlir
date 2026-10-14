// An omp.parallel with an if clause under pmsis: ext_pi_cl_team_fork has no
// `if` parameter, so the clause lowers the way libgomp's does — the false side
// forks a one-core team, which is the serial execution.  It stays a fork rather
// than a direct call to the closure so the region keeps running inside a team:
// an ext_pi_cl_team_barrier in the body waits for the cores of the team it was
// forked with, and a body running outside one would never meet that count.
//
// With no num_threads clause the true side passes `default_team_size` (8); with
// one, the clause value.  That also exercises a ConstructOp carrying two named
// clause operands at once (num_threads + if_clause).
//
// This test used to assert a warning instead: pmsis had no `if` lowering and
// the outlining pass reported the clause as ignored.  See
// parallel-if-{iomp,libgomp}.mlir for the same property on the other runtimes.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @parallel_if(%arg0: !llvm.ptr, %cond: i1) {
  omp.parallel if(%cond) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

func.func @parallel_if_nt(%arg0: !llvm.ptr, %cond: i1, %nt: i32) {
  omp.parallel if(%cond) num_threads(%nt : i32) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// Without num_threads: the default team size on the true side, one core on the
// false one.
// CHECK-LABEL: func.func @parallel_if
// CHECK:         llvm.cond_br %{{.*}}, ^[[T:bb[0-9]+]], ^[[F:bb[0-9]+]]
// CHECK:       ^[[T]]:
// CHECK-DAG:     %[[EIGHT:.*]] = arith.constant 8 : i32
// CHECK:         call @ext_pi_cl_team_fork(%[[EIGHT]], %{{.*}}, %{{.*}})
// CHECK:         llvm.br ^[[J:bb[0-9]+]]
// CHECK:       ^[[F]]:
// CHECK-DAG:     %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:         call @ext_pi_cl_team_fork(%[[ONE]], %{{.*}}, %{{.*}})
// CHECK:         llvm.br ^[[J]]

// With num_threads(%nt): the clause value on the true side, one core on the
// false one.
// CHECK-LABEL: func.func @parallel_if_nt
// CHECK-SAME:    %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: i1, %[[NT:[^:]+]]: i32
// CHECK:         llvm.cond_br %{{.*}}, ^[[T2:bb[0-9]+]], ^[[F2:bb[0-9]+]]
// CHECK:       ^[[T2]]:
// CHECK:         call @ext_pi_cl_team_fork(%[[NT]], %{{.*}}, %{{.*}})
// CHECK:         llvm.br ^[[J2:bb[0-9]+]]
// CHECK:       ^[[F2]]:
// CHECK-DAG:     %[[ONE2:.*]] = arith.constant 1 : i32
// CHECK:         call @ext_pi_cl_team_fork(%[[ONE2]], %{{.*}}, %{{.*}})
// CHECK:         llvm.br ^[[J2]]
