// Outlines omp_lower.construct body regions into func.func ops and lowers
// omp.wsloop using the DSL rules.

#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
std::unique_ptr<mlir::Pass> createOmpOutliningPass(
    std::string dslFile = "rules.dsl",
    std::string runtime  = "iomp");
} // namespace mlir
