// OmpOutliningPass.h
//
// Pass that outlines omp_lower.construct body regions into func.func ops
// and lowers omp.wsloop using the DSL rules.
//
// Uses the same --omp-lower-dsl and --omp-lower-runtime flags as
// OmpToOmpLowerPass.

#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
std::unique_ptr<mlir::Pass> createOmpOutliningPass(
    std::string dslFile = "rules.dsl",
    std::string runtime  = "iomp");
void registerOmpOutliningPass();
} // namespace mlir
