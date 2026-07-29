# DSL first: it is the one library the other two build on, and the only one
# that knows nothing about MLIR.
add_subdirectory(DSL)
add_subdirectory(IR)
add_subdirectory(Transforms)
