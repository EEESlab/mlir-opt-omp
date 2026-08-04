// A task with an EXPLICIT firstprivate clause (privatizer recipe with a copy
// region) under libgomp.  libgomp reuses the packed/closure privatizer path it
// shares with omp.parallel: the firstprivate value is copied into a task-private
// alloca inside the outlined body, and the leftover privatizer block arg is
// dropped — so the outlined function stays a single-ptr closure.  The
// firstprivate must NOT leak in as an extra parameter, which would break the
// void(void*) closure ABI the runtime calls.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

omp.private {type = firstprivate} @fp_i32 : i32 copy {
^bb0(%src: !llvm.ptr, %dst: !llvm.ptr):
  %v = llvm.load %src : !llvm.ptr -> i32
  llvm.store %v, %dst : i32, !llvm.ptr
  omp.yield(%dst : !llvm.ptr)
}

llvm.func @use(i32)

func.func @task_fp(%arg0: !llvm.ptr) {
  omp.task private(@fp_i32 %arg0 -> %p : !llvm.ptr) {
    %v = llvm.load %p : !llvm.ptr -> i32
    llvm.call @use(%v) : (i32) -> ()
    omp.terminator
  }
  return
}

// Outlined into a single-ptr closure — the firstprivate arg is NOT leaked into
// the signature (anchoring on `(` + forbidding `,`/`)` in the arg forces exactly
// one parameter):
// CHECK: func.func {{.*}}@outlined_task_{{[0-9]+}}(%{{[^,)]*}}: !llvm.ptr)
// The firstprivate value is copied into a task-private alloca inside the body:
// CHECK:   llvm.alloca
// CHECK:   call @GOMP_task
