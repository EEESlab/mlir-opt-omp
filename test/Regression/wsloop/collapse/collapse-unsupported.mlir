// A collapsed loop nest — `omp.loop_nest (%i, %j)`, from `collapse(2)` — is not
// supported: only the outermost dimension is distributed and rewired, so the
// inner iteration space would silently go missing.
//
// It used to be worse than missing. The body is moved out of the nest region
// while its uses of the inner induction variables are still live, and erasing
// the op then destroys values that are still referenced: the tool aborted on an
// MLIR assertion rather than saying anything. This pins the diagnostic.
//
// One RUN line covers all three runtimes because the check sits ahead of
// anything the rules decide.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

llvm.func @use(i32)

func.func @two_dims() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 4 : i32
    %step = arith.constant 1 : i32
    // expected-error @below {{a collapsed loop nest (2 dimensions) is not supported}}
    omp.wsloop {
      omp.loop_nest (%i, %j) : i32 = (%lb, %lb) to (%ub, %ub) step (%step, %step) {
        llvm.call @use(%j) : (i32) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
