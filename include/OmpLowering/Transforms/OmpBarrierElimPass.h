// OmpBarrierElimPass.h
//
// Declares the MLIR pass that removes team barriers made redundant by the
// surrounding OpenMP structure.  It runs on the omp dialect, before a runtime
// has been chosen.
//
// Registration:
//   #include "OmpLowering/Transforms/OmpBarrierElimPass.h"
//   mlir::registerOmpBarrierElimPass();
//
// Usage in a pass pipeline string:
//   mlir-opt-omp --omp-barrier-elim input.mlir

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<mlir::Pass> createOmpBarrierElimPass();
void registerOmpBarrierElimPass();

} // namespace mlir
