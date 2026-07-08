// End-to-end smoke test for EXPLICIT firstprivate on a task.
//
// The task takes `x` (== 42) firstprivate via a privatizer recipe with a copy
// region, then writes its private copy out through a shared pointer.  The
// printed result is 42 iff the firstprivate copy-in actually ran: the task body
// reads the privatizer block arg (%px), which is only initialised if the value
// was copied from the source at task creation.  If the copy-in is missing the
// read observes uninitialised memory (not 42) or the program crashes.
//
// This is the runtime counterpart of the regression tests
// task-firstprivate-{libgomp,iomp}.mlir:
//   - libgomp reuses the packed/closure privatizer copy-in → prints 42.
//   - iomp v1 has NO firstprivate wiring in outlineTaskShareds (silent ABI
//     break); run_tasks.sh treats this as a KNOWN GAP, not a hard failure.
//
// Top-level task (no enclosing parallel) to keep the capture path minimal; it
// runs in the initial implicit team and completes at the taskwait.  %pout is a
// by-value pointer capture, so the store lands in main's `out`.

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
    %x    = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr       // firstprivate source
    %out  = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr       // task writes here
    %pout = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
    %c42  = llvm.mlir.constant(42 : i32) : i32
    %c0   = llvm.mlir.constant(0 : i32) : i32
    llvm.store %c42, %x : i32, !llvm.ptr                     // x = 42
    llvm.store %c0, %out : i32, !llvm.ptr
    llvm.store %out, %pout : !llvm.ptr, !llvm.ptr            // pout = &out

    omp.task private(@fp_i32 %x -> %px : !llvm.ptr) {
      %v  = llvm.load %px : !llvm.ptr -> i32                 // firstprivate copy (== 42)
      %pp = llvm.load %pout : !llvm.ptr -> !llvm.ptr
      llvm.store %v, %pp : i32, !llvm.ptr                    // *out = 42
      omp.terminator
    }
    omp.taskwait                                             // complete the task

    %r    = llvm.load %out : !llvm.ptr -> i32
    %fmt  = llvm.mlir.addressof @fmt : !llvm.ptr
    %call = llvm.call @printf(%fmt, %r) vararg(!llvm.func<i32 (ptr, ...)>)
              : (!llvm.ptr, i32) -> i32
    %ret  = llvm.mlir.constant(0 : i32) : i32
    func.return %ret : i32
  }
}
