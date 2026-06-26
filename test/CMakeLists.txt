# The omp-lower regression test suite (lit + FileCheck).
#
# Run with:   ninja check-omp        (from the build directory)
# or:         llvm-lit -sv test/

# The mlir-opt-omp executable is emitted in the build root (we deliberately do
# not override LLVM_RUNTIME_OUTPUT_INTDIR, so the compile-*.sh scripts that
# reference ../BUILD/mlir-opt-omp keep working).
set(OMP_LOWER_TOOLS_DIR ${CMAKE_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

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
