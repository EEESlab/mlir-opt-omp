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

// Creates the pass.
std::unique_ptr<mlir::Pass> createPlanLoweringPass();

// Registers the pass with the MLIR pass manager infrastructure so it can be
// referenced by name on the command line.
void registerPlanLoweringPass();

} // namespace mlir
