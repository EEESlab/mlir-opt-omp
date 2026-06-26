cmake_minimum_required(VERSION 3.20)
project(OmpLoweringDialect C CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Default options passed to `lit` by the check-omp target.
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# ---------------------------------------------------------------------------
# TableGen
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS OmpLoweringOps.td)
mlir_tablegen(OmpLoweringOps.dialect.h.inc    -gen-dialect-decls)
mlir_tablegen(OmpLoweringOps.dialect.cpp.inc  -gen-dialect-defs)
mlir_tablegen(OmpLoweringAttrs.h.inc          -gen-attrdef-decls)
mlir_tablegen(OmpLoweringAttrs.cpp.inc        -gen-attrdef-defs)
mlir_tablegen(OmpLoweringOps.h.inc            -gen-op-decls)
mlir_tablegen(OmpLoweringOps.cpp.inc          -gen-op-defs)
add_public_tablegen_target(OmpLoweringOpsIncGen)

# ---------------------------------------------------------------------------
# Dialect + pass static library
# ---------------------------------------------------------------------------
add_library(OmpLoweringStatic STATIC
  OmpLoweringOps.cpp
  DSLParser.cpp
  DSLEvaluator.cpp
  OmpOutliningPass.cpp
  OmpToOmpLowerPass.cpp
  PlanLoweringPass.cpp
)
add_dependencies(OmpLoweringStatic OmpLoweringOpsIncGen)
llvm_update_compile_flags(OmpLoweringStatic)
target_include_directories(OmpLoweringStatic PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# ---------------------------------------------------------------------------
# Custom mlir-opt executable with our dialect compiled in
# ---------------------------------------------------------------------------
add_llvm_executable(mlir-opt-omp
  mlir-opt-omp.cpp
  PARTIAL_SOURCES_INTENDED
)
llvm_update_compile_flags(mlir-opt-omp)

target_include_directories(mlir-opt-omp PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${LLVM_INSTALL_PREFIX}/include
  #  /home/tagliavini/MLIR/BUILD-LLVM/tools/clang/include  # generated CIR headers
)

target_link_libraries(mlir-opt-omp PRIVATE
  OmpLoweringStatic

  # CIR dialect libraries — must come from the BUILD tree because the
  # install does not contain the generated CIRDialect constructor.
  ${LLVM_INSTALL_PREFIX}/lib/libMLIRCIR.a
  ${LLVM_INSTALL_PREFIX}/lib/libMLIRCIRInterfaces.a
  ${LLVM_INSTALL_PREFIX}/lib/libMLIRCIRTransforms.a
  ${LLVM_INSTALL_PREFIX}/lib/libclangCIR.a
  ${LLVM_INSTALL_PREFIX}/lib/libclangCIRLoweringDirectToLLVM.a
  ${LLVM_INSTALL_PREFIX}/lib/libclangCIRLoweringCommon.a
  ${LLVM_INSTALL_PREFIX}/lib/libclangCIRFrontendAction.a

  # Clang support libs that CIR depends on
  ${LLVM_INSTALL_PREFIX}/lib/libclangAST.a
  ${LLVM_INSTALL_PREFIX}/lib/libclangBasic.a

  # mlir-opt infrastructure
  MLIROptLib

  # Dialects
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRBytecodeWriter

  # Pass infrastructure
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
)

# ---------------------------------------------------------------------------
# Regression test suite (lit + FileCheck)
# ---------------------------------------------------------------------------
set(OMP_LOWER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OMP_LOWER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(test)
