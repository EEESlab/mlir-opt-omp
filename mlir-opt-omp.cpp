// mlir-opt-omp.cpp
//
// Custom mlir-opt with omp_lower dialect and passes compiled in.
//
// Our custom flags --omp-lower-dsl and --omp-lower-runtime are extracted
// from argv BEFORE MlirOptMain sees it, so they don't conflict with its
// internal argument parser.
//
// Usage:
//   ./mlir-opt-omp \
//     --allow-unregistered-dialect \
//     --omp-lower-dsl=rules.dsl \
//     --omp-lower-runtime=iomp \
//     --omp-to-omp-lower \
//     --omp-lower-plan \
//     -o out.mlir in.cir

#include "OmpLoweringOps.h"
#include "OmpOutliningPass.h"
#include "OmpToOmpLowerPass.h"
#include "PlanLoweringPass.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Pre-parse our custom flags out of argv before MlirOptMain sees them.
// Returns the extracted values and a new argv without those flags.
// ---------------------------------------------------------------------------

static void extractCustomFlags(int &argc, char **&argv,
                                std::string &dslFile,
                                std::string &runtime) {
  dslFile = "rules.dsl";
  runtime = "iomp";

  std::vector<char *> newArgv;
  newArgv.push_back(argv[0]); // program name

  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);

    if (arg.starts_with("--omp-lower-dsl=")) {
      dslFile = arg.drop_front(strlen("--omp-lower-dsl=")).str();
    } else if (arg.starts_with("--omp-lower-runtime=")) {
      runtime = arg.drop_front(strlen("--omp-lower-runtime=")).str();
    } else {
      newArgv.push_back(argv[i]);
    }
  }

  // Replace argv with the filtered version.
  // We keep the original array alive; just update argc and the pointer.
  static std::vector<char *> filteredArgv;
  filteredArgv = std::move(newArgv);
  argc = static_cast<int>(filteredArgv.size());
  argv = filteredArgv.data();
}

int main(int argc, char **argv) {
  // Extract our flags before MlirOptMain parses argv.
  std::string dslFile, runtime;
  extractCustomFlags(argc, argv, dslFile, runtime);

  mlir::DialectRegistry registry;
  registry.insert<
    cir::CIRDialect,
    mlir::arith::ArithDialect,
    mlir::func::FuncDialect,
    mlir::LLVM::LLVMDialect,
    mlir::omp::OpenMPDialect,
    mlir::omp_lower::OmpLoweringDialect
  >();

  mlir::registerTransformsPasses();

  // Register passes. The omp-to-omp-lower pass is created with the DSL file
  // and runtime we extracted above.
  mlir::registerPass([&]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createOmpToOmpLowerPass(dslFile, runtime);
  });

  mlir::registerPass([&]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createOmpOutliningPass(dslFile, runtime);
  });
  mlir::registerPlanLoweringPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "omp-lower-opt", registry));
}
