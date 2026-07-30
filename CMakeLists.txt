cmake_minimum_required(VERSION 3.20)
project(OmpLoweringDialect C CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Options passed to lit by the check-omp target.
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

# The binary goes in the build root, not build/bin: that is where OMP_TOOL_BIN
# and the quick-compile/ scripts look for it.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR})
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
# Headers are included by full path ("OmpLowering/IR/..."); the binary tree is
# on the path too, since TableGen writes its .inc files under the same subpath.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# ---------------------------------------------------------------------------
# Optional CIR support
# ---------------------------------------------------------------------------
# The passes run on omp + llvm dialect IR; CIR is linked in only so the tool
# can parse modules that still carry cir.*, i.e. the C front-end path. Without
# it the tool still builds, for hand-written or Flang-produced MLIR.
#
# Probe LLVM_LIBRARY_DIR rather than LLVM_INSTALL_PREFIX: the latter is unset
# when MLIR_DIR points at an LLVM build tree, the usual ClangIR setup.
#
# This only picks the default; option() honours the cache, so a build directory
# configured before CIR existed keeps it off until -DOMP_LOWER_ENABLE_CIR=ON.
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
add_subdirectory(include/OmpLowering)
add_subdirectory(lib)
add_subdirectory(tools)

set(OMP_LOWER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OMP_LOWER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(test)
