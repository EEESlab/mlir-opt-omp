// A parallel region that captures two pointers is outlined into an iomp
// microtask void(ptr gtid, ptr btid, ptr cap0, ptr cap1).  OmpOutliningPass
// marks the capture pointer args `noalias`: each capture is a pointer to a
// distinct caller-side slot, so accesses through one never alias another.  The
// attribute lets the optimiser hoist base-pointer loads out of the outlined
// loop and vectorise the body — without it the microtask body stays scalar.
// gtid/btid (the first two args) are runtime-owned and left untouched.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @sink(!llvm.ptr)

func.func @cap_two_ptrs(%a: !llvm.ptr, %b: !llvm.ptr) {
  omp.parallel {
    llvm.call @sink(%a) : (!llvm.ptr) -> ()
    llvm.call @sink(%b) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// The two capture pointer args carry llvm.noalias; the outlined signature is
// emitted on a single line, so both attributes appear after the func name.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_
// CHECK-SAME:  !llvm.ptr {llvm.noalias}
// CHECK-SAME:  !llvm.ptr {llvm.noalias}
