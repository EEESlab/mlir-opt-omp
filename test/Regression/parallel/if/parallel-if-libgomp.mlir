// An omp.parallel with an if clause under libgomp: GOMP_parallel has no `if`
// parameter, so the clause lowers GCC-style by running the region with a
// one-thread team when the condition is false.  With no num_threads clause the
// true side passes 0 ("runtime default"); with one, the clause value.  This
// also exercises a ConstructOp carrying two named clause operands at once
// (num_threads + if_clause).
//
// The choice is on a runtime value, so rules.dsl states it as a `branch` and
// PlanLoweringPass emits a cond_br with a GOMP_parallel call per arm.  It used
// to be an arith.select built in the outlining pass; both run the region with
// one thread on the false side, but only the branch form is expressed in the
// rules rather than in C++.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
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

// Without num_threads: 0 on the true side, 1 on the false side.
// CHECK-LABEL: func.func @parallel_if
// CHECK:         llvm.cond_br %{{.*}}, ^[[T:bb[0-9]+]], ^[[F:bb[0-9]+]]
// CHECK:       ^[[T]]:
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:         call @GOMP_parallel(%{{.*}}, %{{.*}}, %[[ZERO]], %{{.*}})
// CHECK:         llvm.br ^[[J:bb[0-9]+]]
// CHECK:       ^[[F]]:
// CHECK-DAG:     %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:         call @GOMP_parallel(%{{.*}}, %{{.*}}, %[[ONE]], %{{.*}})
// CHECK:         llvm.br ^[[J]]

// With num_threads(%nt): the clause value on the true side, 1 on the false one.
// CHECK-LABEL: func.func @parallel_if_nt
// CHECK:         llvm.cond_br %{{.*}}, ^[[T2:bb[0-9]+]], ^[[F2:bb[0-9]+]]
// CHECK:       ^[[T2]]:
// CHECK:         call @GOMP_parallel(%{{.*}}, %{{.*}}, %arg2, %{{.*}})
// CHECK:         llvm.br ^[[J2:bb[0-9]+]]
// CHECK:       ^[[F2]]:
// CHECK-DAG:     %[[ONE2:.*]] = arith.constant 1 : i32
// CHECK:         call @GOMP_parallel(%{{.*}}, %{{.*}}, %[[ONE2]], %{{.*}})
// CHECK:         llvm.br ^[[J2]]

// The select the outlining pass used to build is gone.
// CHECK-NOT:   arith.select
