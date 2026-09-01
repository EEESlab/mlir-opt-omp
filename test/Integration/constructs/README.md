# OpenMP MLIR Tests

This directory contains standalone `.mlir` test modules for testing OpenMP constructs and clauses which are not present in the PolyBench kernels.

## What a Run Proves

Running these tests validates the backend dialect lowering, transformation passes, and runtime integration against an expected stdout output (e.g., `42`). 

Execute `./run_constructs.sh` 