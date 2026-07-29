# TableGen for the omp_lower dialect.
#
# The .inc files land in <build>/include/OmpLowering/IR/, mirroring this
# directory, so sources reach them with the same path they use for the
# hand-written headers: #include "OmpLowering/IR/OmpLoweringOps.h.inc".
set(LLVM_TARGET_DEFINITIONS OmpLoweringOps.td)
mlir_tablegen(OmpLoweringOps.dialect.h.inc    -gen-dialect-decls)
mlir_tablegen(OmpLoweringOps.dialect.cpp.inc  -gen-dialect-defs)
mlir_tablegen(OmpLoweringAttrs.h.inc          -gen-attrdef-decls)
mlir_tablegen(OmpLoweringAttrs.cpp.inc        -gen-attrdef-defs)
mlir_tablegen(OmpLoweringOps.h.inc            -gen-op-decls)
mlir_tablegen(OmpLoweringOps.cpp.inc          -gen-op-defs)
add_public_tablegen_target(OmpLoweringOpsIncGen)
