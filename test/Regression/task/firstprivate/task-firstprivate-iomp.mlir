// A task with an EXPLICIT firstprivate clause under iomp.
//
// outlineTaskShareds copies the firstprivate value into a task-private slot
// (mirroring the packed path) and rewrites the privatizer block arg to it, then
// drops the now-dead arg.  The entry therefore keeps the ABI-mandated
// i32(i32 gtid, ptr task) -> i32 signature — no firstprivate parameter leaks
// past `ptr`, which would break the entry the runtime calls with only
// (gtid, task).
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
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

// The entry must be exactly i32(i32 gtid, ptr task) -> i32 — no firstprivate
// parameter leaked past `ptr` (anchoring on `(` + forbidding `,`/`)` inside each
// arg forces exactly two parameters):
// CHECK: func.func {{.*}}@outlined_task_{{[0-9]+}}(%{{[^,)]*}}: i32, %{{[^,)]*}}: !llvm.ptr) -> i32
// The firstprivate value is copied into a task-private alloca in the entry,
// which returns i32:
// CHECK:   llvm.alloca
// CHECK:   return %{{.*}} : i32
