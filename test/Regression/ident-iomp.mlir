// Per-construct ident_t emission on the iomp path (OmpOutliningPass).
//
// An omp.parallel emits a default-flags ident for the fork
//   KMPC                = 0x02 = 2   -> @__omp_ident_2
// An explicit omp.barrier inside the outlined region emits
//   KMPC | BARRIER_EXPL = 0x22 = 34  -> @__omp_ident_22
//
// The ident_t global name encodes the flag bitmask in hex, so asserting the
// name asserts the flags. One private constant global is interned per distinct
// flags value (dedup by construction via lookupSymbol), and they all share a
// single psource string @__omp_src_loc_default = ";unknown;unknown;0;0;;"
// whose length (NUL excluded) is stored in reserved_3 = 22.
// See docs/ident-lowering-spec.md.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=iomp \
// RUN:   --omp-to-omp-lower --omp-outline | FileCheck %s

func.func @parallel_with_barrier() {
  omp.parallel {
    omp.barrier
    omp.terminator
  }
  return
}

// Two distinct ident globals — the hex suffix is the flag bitmask:
// CHECK-DAG: llvm.mlir.global {{.*}}@__omp_ident_2(
// CHECK-DAG: llvm.mlir.global {{.*}}@__omp_ident_22(

// Shared default source-location string (one global, reused by every ident):
// CHECK-DAG: llvm.mlir.global {{.*}}@__omp_src_loc_default(";unknown;unknown;0;0;;

// Struct payload: explicit-barrier flags = 34, reserved_3 = strlen(psource) = 22:
// CHECK-DAG: llvm.mlir.constant(34 : i32)
// CHECK-DAG: llvm.mlir.constant(22 : i32)
// CHECK-DAG: llvm.mlir.addressof @__omp_src_loc_default

// The fork takes the default ident; the barrier takes the explicit-barrier one:
// CHECK-DAG: llvm.mlir.addressof @__omp_ident_2
// CHECK-DAG: llvm.mlir.addressof @__omp_ident_22
// CHECK-DAG: @__kmpc_fork_call
// CHECK-DAG: @__kmpc_barrier
