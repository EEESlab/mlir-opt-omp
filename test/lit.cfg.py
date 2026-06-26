# -*- Python -*-
#
# Main lit configuration for the omp-lower regression test suite.
# Adapted from mlir/examples/standalone/test/lit.cfg.py.

import os

import lit.formats
from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "OMP-LOWER"

# testFormat: shell-based, like the rest of LLVM/MLIR.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: a regression test is a .mlir file with RUN/CHECK annotations.
config.suffixes = [".mlir"]

# test_source_root: the directory holding this config (the `test/` dir).
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: where tests run (inside the build tree).
config.test_exec_root = os.path.join(config.omp_lower_obj_root, "test")

# Things lit should not try to treat as tests.
config.excludes = [
    "Inputs",
    "Integration",
    "CMakeLists.txt",
    "README.md",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

# Make the LLVM tools (FileCheck, count, not, mlir-opt, ...) visible.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.omp_lower_tools_dir, config.llvm_tools_dir]
tools = ["mlir-opt-omp", "mlir-opt", "mlir-translate"]
llvm_config.add_tool_substitutions(tools, tool_dirs)

# %rules_dsl expands to the runtime DSL shipped in the repo root, so tests
# exercise the same rules.dsl that ships with the tool.
config.substitutions.append(
    ("%rules_dsl", os.path.join(config.test_source_root, "..", "rules.dsl"))
)
