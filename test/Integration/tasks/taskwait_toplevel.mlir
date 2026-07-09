// End-to-end test for a TOP-LEVEL taskwait (not inside a parallel).
//
// A top-level task writes 42 through a shared pointer; a top-level omp.taskwait
// then completes it before main reads the value.  This exercises the
// lowerTopLevelLeaf path: a top-level taskwait must resolve a REAL gtid
// (__kmpc_global_thread_num) — under iomp the previous undef-gtid lowering
// crashed here.  libgomp's GOMP_taskwait() takes no args and was already fine.
//
// Prints 42 iff the task ran and the taskwait completed it (no crash).

module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @fmt("%d\0A\00")

  func.func @main() -> i32 {
    %c1  = llvm.mlir.constant(1 : i64) : i64
    %x   = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr
    %p   = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
    %z   = llvm.mlir.constant(0 : i32) : i32
    llvm.store %z, %x : i32, !llvm.ptr
    llvm.store %x, %p : !llvm.ptr, !llvm.ptr          // p = &x

    omp.task {
      %pp = llvm.load %p : !llvm.ptr -> !llvm.ptr     // pp = &x (by-value ptr capture)
      %v  = llvm.mlir.constant(42 : i32) : i32
      llvm.store %v, %pp : i32, !llvm.ptr             // *p = 42
      omp.terminator
    }
    omp.taskwait                                      // top-level: complete the task

    %r   = llvm.load %x : !llvm.ptr -> i32
    %fmt = llvm.mlir.addressof @fmt : !llvm.ptr
    %call = llvm.call @printf(%fmt, %r) vararg(!llvm.func<i32 (ptr, ...)>)
              : (!llvm.ptr, i32) -> i32
    %ret = llvm.mlir.constant(0 : i32) : i32
    func.return %ret : i32
  }
}
