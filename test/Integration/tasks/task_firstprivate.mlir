// End-to-end smoke test for EXPLICIT firstprivate on a task.
//
// A task nested in a parallel takes `x` (== 42) firstprivate via a privatizer
// recipe with a copy region, then writes its private copy out through a shared
// pointer.  The printed result is 42 iff the firstprivate copy-in ran: the task
// body reads the privatizer block arg (%px), which is only initialised if the
// value was copied from the source.  If the copy-in is missing the read
// observes uninitialised memory (not 42) or the program crashes.
//
// The task is nested in a parallel (not top-level) so the parallel's implicit
// barrier completes it before main reads `out` — the same sync as task_nested,
// keeping this test focused on firstprivate.  (Top-level taskwait is exercised
// separately by taskwait_toplevel.mlir.)  %pout is a by-value pointer capture,
// so the store lands in main's `out`; x is unmutated, so every thread writes 42.
//
// libgomp copies in via the packed path; iomp via outlineTaskShareds.

module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @fmt("%d\0A\00")

  omp.private {type = firstprivate} @fp_i32 : i32 copy {
  ^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
    %v = llvm.load %src : !llvm.ptr -> i32
    llvm.store %v, %dst : i32, !llvm.ptr
    omp.yield(%dst : !llvm.ptr)
  }

  func.func @main() -> i32 {
    %c1   = llvm.mlir.constant(1 : i64) : i64
    %x    = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr       // firstprivate source (== 42)
    %out  = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr       // task writes here
    %pout = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
    %c42  = llvm.mlir.constant(42 : i32) : i32
    %c0   = llvm.mlir.constant(0 : i32) : i32
    llvm.store %c42, %x : i32, !llvm.ptr                     // x = 42
    llvm.store %c0, %out : i32, !llvm.ptr
    llvm.store %out, %pout : !llvm.ptr, !llvm.ptr            // pout = &out

    omp.parallel {
      omp.task private(@fp_i32 %x -> %px : !llvm.ptr) {
        %v  = llvm.load %px : !llvm.ptr -> i32               // firstprivate copy (== 42)
        %pp = llvm.load %pout : !llvm.ptr -> !llvm.ptr
        llvm.store %v, %pp : i32, !llvm.ptr                  // *out = 42
        omp.terminator
      }
      omp.terminator
    }

    %r    = llvm.load %out : !llvm.ptr -> i32
    %fmt  = llvm.mlir.addressof @fmt : !llvm.ptr
    %call = llvm.call @printf(%fmt, %r) vararg(!llvm.func<i32 (ptr, ...)>)
              : (!llvm.ptr, i32) -> i32
    %ret  = llvm.mlir.constant(0 : i32) : i32
    func.return %ret : i32
  }
}
