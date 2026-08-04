// An omp.barrier directly inside a task region is invalid OpenMP: a barrier may
// not appear in an explicit task.  The outlining pass diagnoses it rather than
// emitting a stray/ill-formed runtime call.
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --verify-diagnostics

llvm.func @use(!llvm.ptr)

func.func @task_with_barrier(%arg0: !llvm.ptr) {
  omp.parallel {
    omp.task {
      llvm.call @use(%arg0) : (!llvm.ptr) -> ()
      // expected-error @below {{'omp.barrier' is not valid inside a task region}}
      omp.barrier
      omp.terminator
    }
    omp.terminator
  }
  return
}
