// Reads the DSL, walks omp.* ops, evaluates the lowering plan for each and
// emits omp_lower.construct ops.  Runs before PlanLoweringPass.

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {

std::unique_ptr<mlir::Pass>
createOmpToOmpLowerPass(std::string dslFile, std::string runtime);

} // namespace mlir
