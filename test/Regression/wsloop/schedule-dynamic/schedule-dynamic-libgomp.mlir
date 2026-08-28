// schedule(dynamic) under libgomp: the iterations arrive a chunk at a time from
// GOMP_loop_dynamic_start / _next, so the sequential loop gets an outer one
// around it, rotated — the opening call in a guard, the repeat call in a latch.
// libgomp is what forces that rotation: `start` hands over the first chunk as
// well as registering the work-share, so _next cannot stand in for it.
//
// Its three chunk_* properties are all pinned below, since each shows up as a
// different shape of IR: i64 slots (the sext in, the trunc out), an i8 answer,
// and an exclusive bound (slt, where iomp gets sle).
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

llvm.func @work(i32)

func.func @wsloop_dynamic() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop schedule(dynamic) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        llvm.call @work(%iv) : (i32) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// The bounds widen into GOMP's long-based ABI on the way in; the trailing 1 is
// the construct's own `default_chunk`, materialised in that same width.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_0
// CHECK:         llvm.sext {{.*}} : i32 to i64
// CHECK:         %[[CHUNK:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[FIRST:.*]] = call @GOMP_loop_dynamic_start(%{{.*}}, %{{.*}}, %{{.*}}, %[[CHUNK]], %[[PLB:.*]], %[[PUB:.*]]) : (i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i8
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:         %[[WORK:.*]] = llvm.icmp "ne" %[[FIRST]], %[[ZERO]] : i8
// CHECK:         llvm.cond_br %[[WORK]], ^[[CHUNKBODY:bb[0-9]+]], ^[[AFTER:bb[0-9]+]]

// Each chunk comes back to the loop's own width, and the induction variable
// starts at what the runtime wrote into the lower slot.  The comparison is slt:
// `iend` is one past the end.
// CHECK:       ^[[CHUNKBODY]]:
// CHECK-NEXT:    %[[LOW64:.*]] = llvm.load %[[PLB]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[LOW:.*]] = llvm.trunc %[[LOW64]] : i64 to i32
// CHECK-NEXT:    %[[HIGH64:.*]] = llvm.load %[[PUB]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[HIGH:.*]] = llvm.trunc %[[HIGH64]] : i64 to i32
// CHECK-NEXT:    llvm.store %[[LOW]], %[[IV:.*]] : i32, !llvm.ptr
// CHECK:       ^[[HEADER:bb[0-9]+]]:
// CHECK-NEXT:    %[[I:.*]] = llvm.load %[[IV]] : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.icmp "slt" %[[I]], %[[HIGH]] : i32

// And the body runs on that induction variable.
// CHECK:         llvm.call @work(%[[I]])

// The latch asks for the next chunk; the exit is reached both when the loop is
// done and when this thread never got one.
// CHECK:         %[[NEXT:.*]] = call @GOMP_loop_dynamic_next(%[[PLB]], %[[PUB]]) : (!llvm.ptr, !llvm.ptr) -> i8
// CHECK:         %[[MORE:.*]] = llvm.icmp "ne" %[[NEXT]], %{{.*}} : i8
// CHECK:         llvm.cond_br %[[MORE]], ^[[CHUNKBODY]], ^[[AFTER]]
// CHECK:       ^[[AFTER]]:
// CHECK:         call @GOMP_loop_end()

func.func @wsloop_dynamic_chunk(%chk: i32) {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop schedule(dynamic = %chk : i32) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        llvm.call @work(%iv) : (i32) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// An explicit chunk takes the default's place: the clause's own value, widened
// like every other index crossing into the ABI.  It reaches the outlined
// function through the closure struct, hence the load.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_1
// CHECK:         %[[CAP:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:         %[[CAP64:.*]] = llvm.sext %[[CAP]] : i32 to i64
// CHECK:         call @GOMP_loop_dynamic_start(%{{.*}}, %{{.*}}, %{{.*}}, %[[CAP64]], %{{.*}}, %{{.*}})

func.func @wsloop_dynamic_nowait() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop nowait schedule(dynamic) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        llvm.call @work(%iv) : (i32) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// nowait does not switch the closing call off, it picks the other one: the
// work-share still has to be released, only the barrier is skipped.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_2
// CHECK:         call @GOMP_loop_dynamic_start
// CHECK-NOT:     call @GOMP_loop_end()
// CHECK:         call @GOMP_loop_end_nowait()
