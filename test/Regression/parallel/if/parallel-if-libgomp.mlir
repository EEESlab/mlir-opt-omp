// An omp.parallel with an if clause under libgomp: GOMP_parallel has no `if`
// parameter, so the clause lowers GCC-style by forcing num_threads to 1 when
// the condition is false (a one-thread team is the serial execution).  With no
// num_threads clause the true side keeps 0 ("runtime default"); with one, the
// clause value.  This also exercises a ConstructOp carrying two named clause
// operands at once (num_threads + if_clause).
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

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

// Without num_threads: select between 0 (default team size) and 1 (serial).
// CHECK-LABEL: func.func @parallel_if
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG:   %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:       %[[NT:.*]] = arith.select %{{.*}}, %[[ZERO]], %[[ONE]] : i32
// CHECK:       call @GOMP_parallel({{.*}}, %[[NT]], %{{.*}})

// With num_threads(%nt): select between the clause value and 1.
// CHECK-LABEL: func.func @parallel_if_nt
// CHECK:       %[[ONE2:.*]] = arith.constant 1 : i32
// CHECK:       %[[NT2:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ONE2]] : i32
// CHECK:       call @GOMP_parallel({{.*}}, %[[NT2]], %{{.*}})
