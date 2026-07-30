# The omp-lower regression test suite (lit + FileCheck).
#
# Run with:   ninja check-omp        (from the build directory)
# or:         llvm-lit -sv test/

# The mlir-opt-omp executable is emitted in the build root: the top-level
# CMakeLists pins LLVM_RUNTIME_OUTPUT_INTDIR there so declaring the target
# under tools/ does not move it, and the Integration configs that point
# OMP_TOOL_BIN at the build directory keep working.
set(OMP_LOWER_TOOLS_DIR ${CMAKE_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# An LLVM install has no llvm-lit, and AddLLVM falls back to our build tree.
if(NOT LLVM_EXTERNAL_LIT AND NOT EXISTS "${LLVM_DEFAULT_EXTERNAL_LIT}")
  find_program(OMP_LOWER_LIT NAMES llvm-lit lit lit.py)
  if(OMP_LOWER_LIT)
    set(LLVM_EXTERNAL_LIT "${OMP_LOWER_LIT}"
        CACHE STRING "Command used to spawn lit" FORCE)
    message(STATUS "mlir-opt-omp: lit at ${OMP_LOWER_LIT}")
  else()
    message(WARNING
      "mlir-opt-omp: no lit found, so the check-omp target will fail. "
      "Install one (pip install lit), or configure with "
      "-DLLVM_EXTERNAL_LIT=<path to llvm-lit> pointing at an LLVM build tree.")
  endif()
endif()

# mlir-opt-omp is always ours to build. FileCheck/count/not are real CMake
# targets only for an in-tree LLVM; against an installed/external LLVM they are
# plain binaries found via PATH, so only depend on them when the target exists.
set(OMP_LOWER_TEST_DEPENDS mlir-opt-omp)
foreach(tool FileCheck count not)
  if(TARGET ${tool})
    list(APPEND OMP_LOWER_TEST_DEPENDS ${tool})
  endif()
endforeach()

add_lit_testsuite(check-omp "Running the omp-lower regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${OMP_LOWER_TEST_DEPENDS}
)
set_target_properties(check-omp PROPERTIES FOLDER "Tests")

add_lit_testsuites(OMP_LOWER ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${OMP_LOWER_TEST_DEPENDS}
)
