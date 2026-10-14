// The __kmpc_* work-sharing entry points take the upper bound as the last valid
// iteration, and omp.loop_nest gives it as one past the end.  Converting one to
// the other is not a subtraction of the step: the step need not divide the
// range.  The loop below runs 0, 3, 6, 9 and ends at 9, where ub - step is 7 —
// and 7 is what the trip count would then be computed from, so the runtime
// would hand out three iterations instead of four and 9 would run nowhere.
//
// Both iomp constructs read that bound, so both are checked here.  The other
// two runtimes are unaffected: libgomp's loop_start takes the exclusive bound
// as it stands, and the inline distribution rounds the trip count up itself.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

func.func @static_step3() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 3 : i32
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// The bound is derived from the trip count — rounded up, then walked back one
// step from the end — and it is that value the slot is seeded with.  A
// regression to `ub - step` would leave a lone llvm.sub feeding the store.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_0
// CHECK:         %[[TRIP:.*]] = llvm.sdiv
// CHECK:         %[[LAST:.*]] = llvm.sub %[[TRIP]], %{{.*}} : i32
// CHECK:         %[[OFF:.*]] = llvm.mul %[[LAST]], %{{.*}} : i32
// CHECK:         %[[UBI:.*]] = llvm.add %{{.*}}, %[[OFF]] : i32
// CHECK:         llvm.store %[[UBI]], %[[PUB:.*]] : i32, !llvm.ptr
// CHECK:         call @__kmpc_for_static_init_4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PUB]], %{{.*}}, %{{.*}}, %{{.*}})

func.func @dynamic_step3() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 3 : i32
    omp.wsloop schedule(dynamic) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// Same value, passed by value this time rather than through the slot.
// CHECK-LABEL: func.func {{.*}}@outlined_parallel_1
// CHECK:         %[[TRIP2:.*]] = llvm.sdiv
// CHECK:         %[[LAST2:.*]] = llvm.sub %[[TRIP2]], %{{.*}} : i32
// CHECK:         %[[OFF2:.*]] = llvm.mul %[[LAST2]], %{{.*}} : i32
// CHECK:         %[[UBI2:.*]] = llvm.add %{{.*}}, %[[OFF2]] : i32
// CHECK:         call @__kmpc_dispatch_init_4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[UBI2]], %{{.*}}, %{{.*}})
