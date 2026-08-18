// A `private` whose `init` region does something — here, zeroing the slot.
// Lowering a private means allocating storage, and that is all wirePrivatizers
// emits, so an init region with a body asks for more than it provides.  The
// bare alloca would not be an incomplete lowering but a wrong one, silently
// skipping code the recipe says must run, so the pass refuses the input.
//
// The contrast is private-trivial-init-iomp.mlir, where the region hands the
// slot straight back and asks for nothing.  What separates them is whether the
// region has a body, not whether it exists.
//
// One runtime only: the check sits in OmpToOmpLowerPass, ahead of any
// runtime-specific work.
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
  // expected-error @+1 {{`private` clause whose recipe has a non-trivial init or dealloc region is not supported}}
  omp.parallel private(@priv_init %a -> %p : !llvm.ptr) {
    %v = llvm.load %p : !llvm.ptr -> i32
    llvm.call @use(%v) : (i32) -> ()
    omp.terminator
  }
  return
}
