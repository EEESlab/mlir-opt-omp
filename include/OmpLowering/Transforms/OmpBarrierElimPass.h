// Removes team barriers made redundant by the surrounding OpenMP structure.
// Runs on the omp dialect, before a runtime has been chosen.

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<mlir::Pass> createOmpBarrierElimPass();
void registerOmpBarrierElimPass();

} // namespace mlir
