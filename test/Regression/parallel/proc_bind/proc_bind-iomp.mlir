// proc_bind on iomp reaches the plan as the *string* "close" (see
// extractParallelContext in OmpToOmpLowerPass.cpp), and the emission side
// resolves an unknown string token to `llvm.mlir.undef : !llvm.ptr`.  So
// __kmpc_push_proc_bind — whose third parameter is a kmp_proc_bind_t enum, an
// i32 — is currently handed an undef pointer.  The call is emitted, which is
// why nothing fails loudly today; the argument is simply garbage of the wrong
// type.
//
// The check pins the type the ABI requires rather than the enum value, so it
// stays valid whichever constant close maps to.  It fails today (the printed
// callee type ends in `!llvm.ptr`), hence the XFAIL.
//
// XFAIL: *
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @parallel_pb(%arg0: !llvm.ptr) {
  omp.parallel proc_bind(close) {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// The push must precede the fork (it configures the *next* fork, exactly like
// __kmpc_push_num_threads) and take (ident, gtid, i32 enum).
// CHECK-LABEL: func.func @parallel_pb
// CHECK:         call @__kmpc_push_proc_bind({{.*}}) : (!llvm.ptr, i32, i32) -> ()
// CHECK:         llvm.call @__kmpc_fork_call
