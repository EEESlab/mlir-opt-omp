// End-to-end test for the parallel `if` clause with a FALSE condition: the
// region must run SERIALIZED — a team of one, the encountering thread.  The
// body increments *px by 42; serialized execution means exactly one increment
// (x == 42), while a real fork with OMP_NUM_THREADS=2 would run the body twice
// (x == 84, module the race).
//
//   iomp    → runtime branch: __kmpc_serialized_parallel / direct microtask
//             call / __kmpc_end_serialized_parallel on the false side
//             (__kmpc_fork_call_if is unusable with by_pointer captures: it
//             takes a single packed void*)
//   libgomp → GOMP_parallel with num_threads forced to 1 (GCC-style)
//
// Before this lowering, iomp emitted __kmpc_fork_call_if with the condition
// argument MISSING (captures shifted into its slot) — a miscompile; libgomp
// silently ignored the clause (x would be 84 with 2 threads).

module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @fmt("%d\0A\00")

  func.func @main() -> i32 {
    %c1  = llvm.mlir.constant(1 : i64) : i64
    %x   = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr        // incremented in-region
    %px  = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
    %z   = llvm.mlir.constant(0 : i32) : i32
    llvm.store %z, %x : i32, !llvm.ptr
    llvm.store %x, %px : !llvm.ptr, !llvm.ptr                // px = &x
    %false = llvm.mlir.constant(false) : i1

    omp.parallel if(%false) {
      %ppx = llvm.load %px : !llvm.ptr -> !llvm.ptr          // &x (by-value ptr capture)
      %old = llvm.load %ppx : !llvm.ptr -> i32
      %v   = llvm.mlir.constant(42 : i32) : i32
      %new = llvm.add %old, %v : i32
      llvm.store %new, %ppx : i32, !llvm.ptr                 // *px += 42
      omp.terminator
    }

    %r   = llvm.load %x : !llvm.ptr -> i32                   // 42 iff serialized
    %fmt = llvm.mlir.addressof @fmt : !llvm.ptr
    %call = llvm.call @printf(%fmt, %r) vararg(!llvm.func<i32 (ptr, ...)>)
              : (!llvm.ptr, i32) -> i32
    %ret = llvm.mlir.constant(0 : i32) : i32
    func.return %ret : i32
  }
}
