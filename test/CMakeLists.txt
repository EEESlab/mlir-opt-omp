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

set(OMP_LOWER_TEST_DEPENDS
  FileCheck count not
  mlir-opt-omp
)

add_lit_testsuite(check-omp "Running the omp-lower regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${OMP_LOWER_TEST_DEPENDS}
)
set_target_properties(check-omp PROPERTIES FOLDER "Tests")

add_lit_testsuites(OMP_LOWER ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${OMP_LOWER_TEST_DEPENDS}
)
