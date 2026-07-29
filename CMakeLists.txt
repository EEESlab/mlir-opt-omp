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

# Put the executable in the build root even though it is declared under
# tools/. The Integration configs point OMP_TOOL_BIN at the build directory
# and the quick-compile/ scripts fall back to <repo>/build, so the binary has
# to stay where it has always been.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR})
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
# Sources include their own headers by full path ("OmpLowering/IR/...").
# The binary tree is on the path too: TableGen writes the generated .inc files
# to <build>/include/OmpLowering/IR/, so they are reached the same way.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

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
#
# The link itself happens in tools/mlir-opt-omp/, the only target that needs it.
if(EXISTS "${LLVM_LIBRARY_DIR}/libMLIRCIR.a")
  set(OMP_LOWER_CIR_DEFAULT ON)
else()
  set(OMP_LOWER_CIR_DEFAULT OFF)
endif()
option(OMP_LOWER_ENABLE_CIR
  "Link the ClangIR dialect so modules containing cir.* can be parsed"
  ${OMP_LOWER_CIR_DEFAULT})

if(OMP_LOWER_ENABLE_CIR)
  message(STATUS "mlir-opt-omp: CIR support ON")
else()
  message(STATUS "mlir-opt-omp: CIR support OFF")
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
# include/ only runs TableGen; the libraries live in lib/, the executable in
# tools/, the lit suite in test/.
add_subdirectory(include/OmpLowering)
add_subdirectory(lib)
add_subdirectory(tools)

set(OMP_LOWER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OMP_LOWER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(test)
