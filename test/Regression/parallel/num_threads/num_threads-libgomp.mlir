// libgomp has no push call: num_threads is argument 2 of GOMP_parallel, so the
// clause selects between the `when has(num_threads)` and `otherwise` arms of the
// DSL invoke.  Without the clause the slot is the literal 0, which libgomp reads
// as "runtime default team size".
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
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

// With the clause: the SSA value reaches GOMP_parallel's num_threads slot.
// CHECK-LABEL: func.func @parallel_nt
// CHECK:         call @GOMP_parallel({{.*}}%arg1{{.*}})

// Without it: the default-team-size 0 constant goes in instead.
// CHECK-LABEL: func.func @parallel_plain
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:         call @GOMP_parallel({{.*}}%[[ZERO]]{{.*}})
