// End-to-end smoke test for omp.task lowering: a task nested inside a parallel
// region writes a value through a shared pointer. After the parallel region's
// implicit barrier (which also completes outstanding tasks) main observes 42.
//
// The shared output is a *pointer to* the int (int *p = &x; *p = 42), not the
// int itself: this tool classifies a scalar alloca whose first in-region use is
// a store as a private (per-thread) capture, which would hide the write. A
// pointer capture (first use is a load) is packed by value and the store lands
// in the caller's x.
//
// Lowered through mlir-opt-omp (libgomp) and run against real libgomp by
// run_tasks.sh.

module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @fmt("%d\0A\00")

  func.func @main() -> i32 {
    %c1  = llvm.mlir.constant(1 : i64) : i64
    %x   = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr
    %p   = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
    %z   = llvm.mlir.constant(0 : i32) : i32
    llvm.store %z, %x : i32, !llvm.ptr
    llvm.store %x, %p : !llvm.ptr, !llvm.ptr      // p = &x

    omp.parallel {
      omp.task {
        %pp = llvm.load %p : !llvm.ptr -> !llvm.ptr   // pp = &x (by-value ptr capture)
        %v  = llvm.mlir.constant(42 : i32) : i32
        llvm.store %v, %pp : i32, !llvm.ptr           // *p = 42
        omp.terminator
      }
      omp.terminator
    }

    %r   = llvm.load %x : !llvm.ptr -> i32
    %fmt = llvm.mlir.addressof @fmt : !llvm.ptr
    %call = llvm.call @printf(%fmt, %r) vararg(!llvm.func<i32 (ptr, ...)>)
              : (!llvm.ptr, i32) -> i32
    %ret = llvm.mlir.constant(0 : i32) : i32
    func.return %ret : i32
  }
}
