add_llvm_executable(mlir-opt-omp
  mlir-opt-omp.cpp
  PARTIAL_SOURCES_INTENDED
)
llvm_update_compile_flags(mlir-opt-omp)

# ---------------------------------------------------------------------------
# Optional CIR support
# ---------------------------------------------------------------------------
# OMP_LOWER_ENABLE_CIR is decided at the top level; only this target acts on
# it, since registering the dialect is all the CIR link buys us.
if(OMP_LOWER_ENABLE_CIR)
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
endif()

target_link_libraries(mlir-opt-omp PRIVATE
  # Our libraries. Transforms pulls in the dialect and the DSL through its
  # PUBLIC link deps; the dialect is named again because main() registers it.
  MLIROmpLoweringTransforms
  MLIROmpLowering

  # mlir-opt infrastructure
  MLIROptLib
  MLIRBytecodeWriter
)
