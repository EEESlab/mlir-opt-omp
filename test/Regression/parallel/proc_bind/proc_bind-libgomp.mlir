// proc_bind under libgomp.  GOMP has no per-region push call, so this looked
// like a clause the runtime could not honour — but it can: the fourth argument
// of GOMP_parallel is a flags word carrying the affinity policy in its low
// bits, and GCC passes exactly the numbering iomp uses (close = 3, spread = 4,
// no clause = 0).  Compare `gcc -fopenmp -S` on a region with proc_bind(close):
// the fourth argument register is loaded with 3.
//
// So the clause is lowered here rather than diagnosed, using the always-valued
// `proc_bind_flags` token: unlike iomp's push call there is nothing to gate, the
// slot is passed on every fork and 0 is what "no policy" means in it.
//
// This test used to XFAIL asserting a warning that the clause was ignored,
// which is what the lowering did before — silently, since libgomp's rules never
// mentioned it.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @parallel_close(%arg0: !llvm.ptr) {
  omp.parallel proc_bind(close) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

func.func @parallel_spread(%arg0: !llvm.ptr) {
  omp.parallel proc_bind(spread) {
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

// close -> 3 in the flags word, with the team size left at the 0 that means
// "runtime default".  Binding the constant is what makes this a real check:
// the flags slot holding *some* value would also match an undef or a stray 0.
// CHECK-LABEL: func.func @parallel_close
// CHECK-DAG:     %[[CLOSE:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         call @GOMP_parallel(%{{.*}}, %{{.*}}, %{{.*}}, %[[CLOSE]]) : (!llvm.ptr, !llvm.ptr, i32, i32) -> ()

// spread -> 4.  Two kinds are checked because the MLIR enum's own ordinals
// (close = 2, spread = 3) are a plausible wrong answer that one value alone
// would not catch.
// CHECK-LABEL: func.func @parallel_spread
// CHECK-DAG:     %[[SPREAD:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:         call @GOMP_parallel(%{{.*}}, %{{.*}}, %{{.*}}, %[[SPREAD]])

// No clause: the flags word is still passed, holding 0.
// CHECK-LABEL: func.func @parallel_plain
// CHECK-DAG:     %[[NONE:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         call @GOMP_parallel(%{{.*}}, %{{.*}}, %{{.*}}, %[[NONE]])
