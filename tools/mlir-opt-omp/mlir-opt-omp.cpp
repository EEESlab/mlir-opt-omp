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
//     -o out.mlir in.mlir

#include "OmpLowering/IR/OmpLoweringOps.h"
#include "OmpLowering/Transforms/OmpBarrierElimPass.h"
#include "OmpLowering/Transforms/OmpOutliningPass.h"
#include "OmpLowering/Transforms/OmpToOmpLowerPass.h"
#include "OmpLowering/Transforms/PlanLoweringPass.h"

// CIR is optional: the passes below never look at cir.* operations, the dialect
// is registered only so modules coming straight from the C front-end still
// parse. Configure with -DOMP_LOWER_ENABLE_CIR=OFF to build against a stock
// LLVM/MLIR and feed the tool MLIR that carries no cir.*.
#ifdef OMP_LOWER_HAS_CIR
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#endif

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

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

    if (arg.consume_front("--omp-lower-dsl=")) {
      dslFile = arg.str();
    } else if (arg.consume_front("--omp-lower-runtime=")) {
      runtime = arg.str();
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
    mlir::arith::ArithDialect,
    mlir::func::FuncDialect,
    mlir::LLVM::LLVMDialect,
    mlir::omp::OpenMPDialect,
    mlir::omp_lower::OmpLoweringDialect
  >();
#ifdef OMP_LOWER_HAS_CIR
  registry.insert<cir::CIRDialect>();
#endif

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
  // No DSL file: it optimises the omp dialect
  mlir::registerOmpBarrierElimPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "omp-lower-opt", registry));
}
