// MLIR twin of test.c — the same @add(int a[], int b[], int c[], int n) doing
// `#pragma omp parallel for` over c[i] = a[i] + b[i], written directly in the
// dialects mlir-opt-omp consumes: omp.* over memory already in the llvm
// dialect. Nothing here comes from (or needs) the CIR front-end.
//
// The signature matches test.c, so compile-from-mlir.sh links it against the same
// main.c and the binary must print the same numbers as ./test-ref.
//
// Note the shape the passes look for: each parameter is spilled to an
// llvm.alloca in the entry block and re-loaded inside the parallel region.
// That is what the C front-end emits, and it is how captures are classified —
// a, b and c are ptr allocas read inside the region (captured by value), n is
// a scalar alloca read inside the region (also captured by value). See
// OmpOutliningPass.cpp.

module {
  func.func @add(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %n: i32) {
    %one = llvm.mlir.constant(1 : i64) : i64
    %pa = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
    %pb = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
    %pc = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
    %pn = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    llvm.store %a, %pa : !llvm.ptr, !llvm.ptr
    llvm.store %b, %pb : !llvm.ptr, !llvm.ptr
    llvm.store %c, %pc : !llvm.ptr, !llvm.ptr
    llvm.store %n, %pn : i32, !llvm.ptr

    omp.parallel {
      %lb   = llvm.mlir.constant(0 : i32) : i32
      %step = llvm.mlir.constant(1 : i32) : i32
      %ub   = llvm.load %pn : !llvm.ptr -> i32
      omp.wsloop {
        omp.loop_nest (%i) : i32 = (%lb) to (%ub) step (%step) {
          %va = llvm.load %pa : !llvm.ptr -> !llvm.ptr
          %vb = llvm.load %pb : !llvm.ptr -> !llvm.ptr
          %vc = llvm.load %pc : !llvm.ptr -> !llvm.ptr
          %ga = llvm.getelementptr %va[%i] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          %gb = llvm.getelementptr %vb[%i] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          %gc = llvm.getelementptr %vc[%i] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          %x  = llvm.load %ga : !llvm.ptr -> i32
          %y  = llvm.load %gb : !llvm.ptr -> i32
          %s  = llvm.add %x, %y : i32
          llvm.store %s, %gc : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
