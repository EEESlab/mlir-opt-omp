# The three lowering passes. This is where the DSL meets MLIR: the rules
# produce a plan, the passes turn that plan into IR.
#
# DEPENDS on the TableGen target as well as linking the dialect: these sources
# include the generated headers directly.
add_mlir_library(MLIROmpLoweringTransforms
  OmpToOmpLowerPass.cpp
  OmpOutliningPass.cpp
  PlanLoweringPass.cpp
  PlanEmit.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/OmpLowering/Transforms

  DEPENDS
  OmpLoweringOpsIncGen

  LINK_LIBS PUBLIC
  MLIROmpLowering
  OmpLoweringDSL

  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIROpenMPDialect

  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
