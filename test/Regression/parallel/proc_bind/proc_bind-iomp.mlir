// proc_bind under iomp: __kmpc_push_proc_bind configures the *next* fork,
// exactly like __kmpc_push_num_threads, so it belongs in the construct's `pre`
// block and must precede __kmpc_fork_call.
//
// Its third parameter is a kmp_proc_bind_t, an i32 enum — and the value is the
// point of this test.  The clause has no SSA value to carry, so the kind's
// spelling rides on the ConstructOp as an attribute and the outlining pass
// materialises the constant (procBindEnumValue in PlanEmit.cpp).  The MLIR enum
// numbers its own cases primary=0, master=1, close=2, spread=3; the runtimes
// number theirs false=0, true=1, master=2, close=3, spread=4.  Passing the
// ordinal through would therefore turn close into master and spread into close
// — a silent affinity change — which is what the constants checked below rule
// out.  primary and master are the 5.1 rename of one concept and share a value.
//
// This test used to XFAIL for a different reason: the clause reached the plan
// as the *string* "close", and an unknown string token resolves to
// `llvm.mlir.undef : !llvm.ptr`, so the enum slot was handed an undef pointer.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
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

func.func @parallel_primary(%arg0: !llvm.ptr) {
  omp.parallel proc_bind(primary) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// close -> kmp_proc_bind_close, and the push ahead of the fork.
// CHECK-LABEL: func.func @parallel_close
// CHECK-DAG:     %[[CLOSE:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         call @__kmpc_push_proc_bind(%{{.*}}, %{{.*}}, %[[CLOSE]]) : (!llvm.ptr, i32, i32) -> ()
// CHECK:         llvm.call @__kmpc_fork_call

// spread -> kmp_proc_bind_spread.
// CHECK-LABEL: func.func @parallel_spread
// CHECK-DAG:     %[[SPREAD:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:         call @__kmpc_push_proc_bind(%{{.*}}, %{{.*}}, %[[SPREAD]]) : (!llvm.ptr, i32, i32) -> ()
// CHECK:         llvm.call @__kmpc_fork_call

// primary -> kmp_proc_bind_master, the same value the older spelling maps to.
// CHECK-LABEL: func.func @parallel_primary
// CHECK-DAG:     %[[PRIMARY:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         call @__kmpc_push_proc_bind(%{{.*}}, %{{.*}}, %[[PRIMARY]]) : (!llvm.ptr, i32, i32) -> ()
// CHECK:         llvm.call @__kmpc_fork_call
