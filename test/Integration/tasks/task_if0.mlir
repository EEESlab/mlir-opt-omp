// End-to-end test for the task `if` clause with a FALSE condition: OpenMP
// requires the task to run UNDEFERRED — the spawning thread executes it before
// moving past the task construct.  The read-back below therefore needs NO
// taskwait: if(false) itself is the completion guarantee.
//
//   iomp    → runtime branch: __kmpc_omp_task_begin_if0 / direct entry call /
//             __kmpc_omp_task_complete_if0 (the task is allocated either way)
//   libgomp → GOMP_task(..., if_clause=false, ...) runs fn(data) inline
//
// Data flow (per thread): task does *px = 42 undeferred; the region then
// immediately loads *px and copies it into *py; main prints y.  px/py are
// pointer captures (first in-region use is a load) packed by value, so the
// stores land in the caller's x/y — see task_nested.mlir for the rationale.
// Every thread writes the same 42, so OMP_NUM_THREADS > 1 stays deterministic.
//
// Before the if0 lowering the clause was silently dropped (task always
// deferred) and this read raced with the task body: the undeferred guarantee
// is exactly what this test pins down.

module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @fmt("%d\0A\00")

  func.func @main() -> i32 {
    %c1  = llvm.mlir.constant(1 : i64) : i64
    %x   = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr        // task writes here
    %y   = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr        // copy target, printed
    %px  = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
    %py  = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
    %z   = llvm.mlir.constant(0 : i32) : i32
    llvm.store %z, %x : i32, !llvm.ptr
    llvm.store %z, %y : i32, !llvm.ptr
    llvm.store %x, %px : !llvm.ptr, !llvm.ptr                // px = &x
    llvm.store %y, %py : !llvm.ptr, !llvm.ptr                // py = &y

    omp.parallel {
      // Defined inside the region so it is not a capture of the parallel.
      %false = llvm.mlir.constant(false) : i1
      omp.task if(%false) {
        %ppx = llvm.load %px : !llvm.ptr -> !llvm.ptr        // &x (by-value ptr capture)
        %v   = llvm.mlir.constant(42 : i32) : i32
        llvm.store %v, %ppx : i32, !llvm.ptr                 // *px = 42
        omp.terminator
      }
      // NO taskwait: if(false) means the task already completed here.
      %ppx2 = llvm.load %px : !llvm.ptr -> !llvm.ptr
      %val  = llvm.load %ppx2 : !llvm.ptr -> i32             // x == 42 here
      %ppy  = llvm.load %py : !llvm.ptr -> !llvm.ptr
      llvm.store %val, %ppy : i32, !llvm.ptr                 // *py = *px
      omp.terminator
    }

    %r   = llvm.load %y : !llvm.ptr -> i32
    %fmt = llvm.mlir.addressof @fmt : !llvm.ptr
    %call = llvm.call @printf(%fmt, %r) vararg(!llvm.func<i32 (ptr, ...)>)
              : (!llvm.ptr, i32) -> i32
    %ret = llvm.mlir.constant(0 : i32) : i32
    func.return %ret : i32
  }
}
