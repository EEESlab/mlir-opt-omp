// schedule(dynamic) under iomp: __kmpc_dispatch_init_4 once, then a turn of the
// outer loop per chunk asking __kmpc_dispatch_next_4 for the next range.
//
// Read alongside schedule-dynamic-libgomp.mlir: the two share the whole
// lowering and differ only in what rules.dsl says about them.  Here that is
// nothing — no first_chunk (the same call opens and repeats) and none of the
// chunk_* properties (this ABI is what they default to, see README.md).
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
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

// 35 is kmp_sch_dynamic_chunked; the trailing 1 is `default_chunk`, the clause
// having named none.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_0
// CHECK:         %[[SCHED:.*]] = llvm.mlir.constant(35 : i32) : i32
// CHECK:         %[[CHUNK:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         call @__kmpc_dispatch_init_4(%{{.*}}, %{{.*}}, %[[SCHED]], %{{.*}}, %{{.*}}, %{{.*}}, %[[CHUNK]]) : (!llvm.ptr, i32, i32, i32, i32, i32, i32) -> ()

// The guard: a chunk asked for before the loop is entered at all.  Capturing
// the two bound slots here is what lets the CHECKs below tie the loop to them.
// CHECK:         %[[FIRST:.*]] = call @__kmpc_dispatch_next_4(%{{.*}}, %{{.*}}, %{{.*}}, %[[PLB:.*]], %[[PUB:.*]], %{{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[WORK:.*]] = llvm.icmp "ne" %[[FIRST]], %[[ZERO]] : i32
// CHECK:         llvm.cond_br %[[WORK]], ^[[CHUNKBODY:bb[0-9]+]], ^[[AFTER:bb[0-9]+]]

// Each chunk: the induction variable starts at what the runtime wrote into the
// lower slot and the loop ends at the upper one.  Read straight out, with no
// conversion — same width as the loop — and compared with sle, the bound being
// the last valid iteration rather than one past it.
// CHECK:       ^[[CHUNKBODY]]:
// CHECK-NEXT:    %[[LOW:.*]] = llvm.load %[[PLB]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[HIGH:.*]] = llvm.load %[[PUB]] : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.store %[[LOW]], %[[IV:.*]] : i32, !llvm.ptr
// CHECK:       ^[[HEADER:bb[0-9]+]]:
// CHECK-NEXT:    %[[I:.*]] = llvm.load %[[IV]] : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.icmp "sle" %[[I]], %[[HIGH]] : i32

// And the body runs on that induction variable.
// CHECK:         llvm.call @work(%[[I]])

// The latch: the same call again, back for another chunk or out.
// CHECK:         %[[NEXT:.*]] = call @__kmpc_dispatch_next_4
// CHECK:         %[[MORE:.*]] = llvm.icmp "ne" %[[NEXT]], %{{.*}} : i32
// CHECK:         llvm.cond_br %[[MORE]], ^[[CHUNKBODY]], ^[[AFTER]]

// No fini — that one is for `ordered` — and the work-share's implicit barrier
// on its own ident rather than the loop's.
// CHECK:       ^[[AFTER]]:
// CHECK-NOT:     __kmpc_for_static_fini
// CHECK-NOT:     __kmpc_dispatch_fini
// CHECK:         call @__kmpc_barrier

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

// An explicit chunk takes the default's place.  The microtask ABI passes each
// capture as its own trailing argument, so it arrives as one and needs no
// unpacking — where libgomp reads it out of the closure struct and widens it.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_1
// CHECK-SAME:    (%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %[[CAP:.*]]: i32)
// CHECK:         call @__kmpc_dispatch_init_4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[CAP]])

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

// nowait drops the barrier outright here, where libgomp has to swap one call
// for another because its work-share still needs releasing.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_2
// CHECK:         call @__kmpc_dispatch_init_4
// CHECK-NOT:     call @__kmpc_barrier
// CHECK:         return
