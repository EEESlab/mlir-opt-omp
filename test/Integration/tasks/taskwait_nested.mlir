// End-to-end smoke test for omp.taskwait lowering, where the taskwait is
// *load-bearing*: a task writes 42 through a shared pointer, and the code that
// reads that value back runs later in the same parallel region, AFTER the
// taskwait.  Without the taskwait the read may observe the pre-task value (0)
// because the runtime may defer the task; the taskwait forces completion, so
// the read — and the printed result — is deterministically 42.
//
// Data flow (per thread): task does *px = 42; after `omp.taskwait` the region
// loads *px and copies it into *py; main prints y.  px/py are pointer captures
// (first in-region use is a load) packed by value, so the stores land in the
// caller's x/y — see task_nested.mlir for the capture-classification rationale.
//
// With N threads every thread writes the same value (42) to x and y, so the
// output stays deterministic under OMP_NUM_THREADS > 1.
//
// Lowered through mlir-opt-omp and run against the real runtime (libgomp's
// GOMP_taskwait or libomp's __kmpc_omp_taskwait) by run_tasks.sh.

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
      omp.task {
        %ppx = llvm.load %px : !llvm.ptr -> !llvm.ptr        // &x (by-value ptr capture)
        %v   = llvm.mlir.constant(42 : i32) : i32
        llvm.store %v, %ppx : i32, !llvm.ptr                 // *px = 42
        omp.terminator
      }
      omp.taskwait                                           // wait for the task

      // Runs only after the task completed: reads the 42 it wrote.
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
