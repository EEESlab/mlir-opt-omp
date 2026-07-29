# The omp_lower dialect itself: ConstructOp plus the PlanCall/PlanEmit
# attributes. Everything not written out here comes from the generated .inc.
add_mlir_dialect_library(MLIROmpLowering
  OmpLoweringOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/OmpLowering/IR

  DEPENDS
  OmpLoweringOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
