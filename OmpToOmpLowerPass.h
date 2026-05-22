// OmpToOmpLowerPass.h
//
// Pass that reads a DSL file at runtime, walks omp.* ops, evaluates the
// lowering plan for each, and emits omp_lower.construct ops.
// Run BEFORE PlanLoweringPass.
//
// Command-line usage:
//   --omp-to-omp-lower="dsl=rules.dsl runtime=iomp"

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {

std::unique_ptr<mlir::Pass>
createOmpToOmpLowerPass(std::string dslFile, std::string runtime);


} // namespace mlir
