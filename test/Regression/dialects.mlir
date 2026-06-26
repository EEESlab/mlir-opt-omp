// Smoke test: the custom tool builds, runs, and registers our dialect.
// RUN: mlir-opt-omp --show-dialects | FileCheck %s

// CHECK: Available Dialects:
// CHECK-SAME: omp_lower
