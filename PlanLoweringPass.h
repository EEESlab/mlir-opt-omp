// PlanLoweringPass.h
//
// Declares the MLIR pass that walks every omp_lower.construct operation and
// replaces it with concrete func.call / llvm.call operations that target the
// runtime library (iomp or libgomp) described in the plan attributes.
//
// Registration:
//   #include "PlanLoweringPass.h"
//   mlir::registerPlanLoweringPass();  // or add to a PassPipeline
//
// Usage in a pass pipeline string:
//   mlir-opt --omp-lower-plan input.mlir

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<mlir::Pass> createPlanLoweringPass();
void registerPlanLoweringPass();

} // namespace mlir
