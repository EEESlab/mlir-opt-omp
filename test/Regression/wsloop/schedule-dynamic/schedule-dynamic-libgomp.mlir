// schedule(dynamic) under libgomp: the iterations are handed out a chunk at a
// time by GOMP_loop_dynamic_start / GOMP_loop_dynamic_next, so the sequential
// loop gets an outer loop around it, rotated — the opening call in a guard
// ahead of it, the repeat call in a latch below.  libgomp is the runtime that
// forces that rotation: `start` both registers the work-share and hands over
// the first chunk, so it cannot be replaced by `next` on the first turn.
//
// Three things in this lowering are libgomp's own ABI, stated as properties in
// rules.dsl rather than assumed by the pass, and each is pinned below:
//   chunk_index  = i64  the slots are `long *`, whatever the induction variable
//                       is — hence the sext going in and the trunc coming out
//   chunk_result = i8   the two calls answer with a C _Bool
//   chunk_bound  = exclusive   `iend` is the one-past-the-end, so the inner
//                       loop compares with slt (iomp's dispatch writes the last
//                       valid iteration and compares with sle)
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=libgomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

func.func @wsloop_dynamic() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop schedule(dynamic) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// With no chunk on the clause the size is `let default_chunk = 1` from the
// construct, materialised in the index type the ABI asks for.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_0
// CHECK:         llvm.sext {{.*}} : i32 to i64
// CHECK:         %[[CHUNK:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[FIRST:.*]] = call @GOMP_loop_dynamic_start(%{{.*}}, %{{.*}}, %{{.*}}, %[[CHUNK]], %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i8
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:         %[[WORK:.*]] = llvm.icmp "ne" %[[FIRST]], %[[ZERO]] : i8
// CHECK:         llvm.cond_br %[[WORK]], ^[[CHUNKBODY:bb[0-9]+]], ^[[AFTER:bb[0-9]+]]

// Each chunk arrives in the slots and comes back to the loop's own width.
// CHECK:       ^[[CHUNKBODY]]:
// CHECK:         llvm.trunc {{.*}} : i64 to i32
// CHECK:         llvm.trunc {{.*}} : i64 to i32

// The inner loop stops one short of the bound GOMP wrote, then asks for more.
// CHECK:         llvm.icmp "slt"
// CHECK:         %[[NEXT:.*]] = call @GOMP_loop_dynamic_next(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i8
// CHECK:         %[[MORE:.*]] = llvm.icmp "ne" %[[NEXT]], %{{.*}} : i8
// CHECK:         llvm.cond_br %[[MORE]], ^[[CHUNKBODY]], ^[[AFTER]]

// Reached both when the loop is done and when this thread never got a chunk.
// CHECK:       ^[[AFTER]]:
// CHECK:         call @GOMP_loop_end()

func.func @wsloop_dynamic_chunk(%chk: i32) {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop schedule(dynamic = %chk : i32) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// An explicit chunk takes the default's place: the clause's own value, widened
// to the ABI's index type like every other index crossing into it.  It reaches
// the outlined function as a capture, hence the load.
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
