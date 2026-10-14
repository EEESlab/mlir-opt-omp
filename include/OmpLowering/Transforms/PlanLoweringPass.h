// Replaces every omp_lower.construct with the concrete runtime calls named by
// its plan attributes.

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<mlir::Pass> createPlanLoweringPass();
void registerPlanLoweringPass();

} // namespace mlir
