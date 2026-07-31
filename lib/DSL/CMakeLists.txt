# The rule-file front end: lexer, parser and evaluator for rules.dsl.
#
# This library has no MLIR dependency at all — it turns rule text into an AST
# and evaluates it against a plain string context. Keeping that separation
# visible in the build is the point of the split: only Transforms/ bridges the
# rules to actual IR.
add_mlir_library(OmpLoweringDSL
  DSLParser.cpp
  DSLEvaluator.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/OmpLowering/DSL

  LINK_COMPONENTS
  Support
)
