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
# Optional CIR support
# ---------------------------------------------------------------------------
# The passes never look at CIR operations: they run on omp + llvm dialect IR.
# CIR is linked in only so the tool can parse modules that still carry cir.*,
# i.e. the C front-end path. Against a stock LLVM/MLIR install the option turns
# itself off and the tool still builds, for hand-written or Flang-produced MLIR.
#
# Probe LLVM_LIBRARY_DIR, not LLVM_INSTALL_PREFIX: the latter is the install
# prefix, which is unset when MLIR_DIR points at an LLVM *build tree* — the
# usual ClangIR setup, since CIR is often built but never installed. The former
# is set correctly by LLVMConfig.cmake for both build trees and installs.
#
# Note this only picks the *default*. option() honours a cached value, so a
# build directory configured while CIR was undetectable keeps it off; force it
# with -DOMP_LOWER_ENABLE_CIR=ON or configure into a clean directory.
if(EXISTS "${LLVM_LIBRARY_DIR}/libMLIRCIR.a")
  set(OMP_LOWER_CIR_DEFAULT ON)
else()
  set(OMP_LOWER_CIR_DEFAULT OFF)
endif()
option(OMP_LOWER_ENABLE_CIR
  "Link the ClangIR dialect so modules containing cir.* can be parsed"
  ${OMP_LOWER_CIR_DEFAULT})

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
)

if(OMP_LOWER_ENABLE_CIR)
  message(STATUS "mlir-opt-omp: CIR support ON")
  target_compile_definitions(mlir-opt-omp PRIVATE OMP_LOWER_HAS_CIR)

  # Clang headers: mlir-opt-omp.cpp includes clang/CIR/Dialect/IR/CIRDialect.h,
  # which in turn pulls in generated .inc files. A build tree keeps the two
  # apart — the header in the clang source tree, the .inc under
  # tools/clang/include — so ask ClangConfig for both instead of guessing.
  find_package(Clang CONFIG QUIET HINTS "${LLVM_LIBRARY_DIR}/cmake/clang")
  if(NOT Clang_FOUND)
    message(FATAL_ERROR
      "OMP_LOWER_ENABLE_CIR is ON but ClangConfig.cmake was not found "
      "(looked in ${LLVM_LIBRARY_DIR}/cmake/clang). Point MLIR_DIR at an LLVM "
      "built with -DCLANG_ENABLE_CIR=ON, or configure with "
      "-DOMP_LOWER_ENABLE_CIR=OFF to build without the C front-end path.")
  endif()
  target_include_directories(mlir-opt-omp PRIVATE ${CLANG_INCLUDE_DIRS})

  target_link_libraries(mlir-opt-omp PRIVATE
    # CIR dialect libraries — taken from LLVM_LIBRARY_DIR so a build tree works
    # as well as an install (the install often lacks the generated CIRDialect
    # constructor, so the build tree is the common case).
    ${LLVM_LIBRARY_DIR}/libMLIRCIR.a
    ${LLVM_LIBRARY_DIR}/libMLIRCIRInterfaces.a
    ${LLVM_LIBRARY_DIR}/libMLIRCIRTransforms.a
    ${LLVM_LIBRARY_DIR}/libclangCIR.a
    ${LLVM_LIBRARY_DIR}/libclangCIRLoweringDirectToLLVM.a
    ${LLVM_LIBRARY_DIR}/libclangCIRLoweringCommon.a
    ${LLVM_LIBRARY_DIR}/libclangCIRFrontendAction.a

    # Clang support libs that CIR depends on
    ${LLVM_LIBRARY_DIR}/libclangAST.a
    ${LLVM_LIBRARY_DIR}/libclangBasic.a
  )
else()
  message(STATUS "mlir-opt-omp: CIR support OFF — input must be free of cir.*")
endif()

target_link_libraries(mlir-opt-omp PRIVATE
  OmpLoweringStatic

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
