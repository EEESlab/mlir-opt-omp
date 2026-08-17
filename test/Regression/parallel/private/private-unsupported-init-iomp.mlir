// A `private` whose recipe carries an `init` region.  Lowering a private means
// allocating a slot, and that is all wirePrivatizers emits — an `init` (or
// `dealloc`) region says the slot needs more than storage: a Fortran descriptor
// to set up, a destructor to run.  Emitting the bare alloca would not be an
// incomplete lowering but a wrong one, silently skipping code the recipe says
// must run, so the pass refuses the input instead.
//
// Only one runtime is exercised: the check sits in OmpToOmpLowerPass, ahead of
// any runtime-specific work, so the other two would re-test the same code path
// through the same pass.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

omp.private {type = private} @priv_init : i32 init {
^bb0(%mold: !llvm.ptr, %alloc: !llvm.ptr):
  %z = llvm.mlir.constant(0 : i32) : i32
  llvm.store %z, %alloc : i32, !llvm.ptr
  omp.yield(%alloc : !llvm.ptr)
}

llvm.func @use(i32)

func.func @priv_with_init(%a: !llvm.ptr) {
  // expected-error @+1 {{`private` clause whose recipe has an init or dealloc region is not supported}}
  omp.parallel private(@priv_init %a -> %p : !llvm.ptr) {
    %v = llvm.load %p : !llvm.ptr -> i32
    llvm.call @use(%v) : (i32) -> ()
    omp.terminator
  }
  return
}
