// Per-construct ident_t emission for a work-sharing loop on the iomp path
// (OmpOutliningPass::lowerWsloop).
//
// __kmpc_for_static_init_4 / __kmpc_for_static_fini get the work-loop ident
//   KMPC | WORK_LOOP        = 0x202 = 514 -> @__omp_ident_202
// The trailing implicit barrier (loop is NOT nowait) gets
//   KMPC | BARRIER_IMPL_FOR = 0x42  = 66  -> @__omp_ident_42
// The enclosing parallel still forks with the default ident
//   KMPC                    = 0x02  = 2   -> @__omp_ident_2
//
// All idents share the single @__omp_src_loc_default psource (reserved_3 = 22).
// See docs/lowering-specs/ident-lowering-spec.md.
//
// NOTE: if omp.wsloop / omp.loop_nest syntax differs in your pinned LLVM,
// adjust the loop below — the behaviour under test is the emitted ident
// globals, not the loop parsing.
//
// The @sink call after the wsloop keeps the loop's implicit barrier from being
// the region's trailing op, so --omp-barrier-elim could not drop it (that
// removal is exercised by barrier-elim/wsloop-trailing.mlir) and its ident is
// emitted here as intended.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

llvm.func @sink(i32)

func.func @parallel_wsloop() {
  omp.parallel {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant 1 : i32
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    }
    llvm.call @sink(%ub) : (i32) -> ()
    omp.terminator
  }
  return
}

// Three distinct ident globals — the hex suffix is the flag bitmask:
// CHECK-DAG: llvm.mlir.global {{.*}}@__omp_ident_2(
// CHECK-DAG: llvm.mlir.global {{.*}}@__omp_ident_202(
// CHECK-DAG: llvm.mlir.global {{.*}}@__omp_ident_42(

// Shared psource string (reserved_3 = 22):
// CHECK-DAG: llvm.mlir.global {{.*}}@__omp_src_loc_default(";unknown;unknown;0;0;;
// CHECK-DAG: llvm.mlir.constant(514 : i32)
// CHECK-DAG: llvm.mlir.constant(66 : i32)
// CHECK-DAG: llvm.mlir.constant(22 : i32)

// init/fini take the work-loop ident; the implicit barrier its own; fork the default:
// CHECK-DAG: llvm.mlir.addressof @__omp_ident_202 :
// CHECK-DAG: llvm.mlir.addressof @__omp_ident_42 :
// CHECK-DAG: llvm.mlir.addressof @__omp_ident_2 :
// CHECK-DAG: @__kmpc_for_static_init_4
// CHECK-DAG: @__kmpc_for_static_fini
// CHECK-DAG: @__kmpc_barrier
