#!/bin/bash
# =============================================================================
# kernels.sh — the PolyBench kernel lists and kernel-selection helpers.
#
# Sourced by common.sh; not meant to be sourced directly.
# Add/remove kernels here. Paths are relative to $POLYBENCH.
# =============================================================================

# The PolyBench/OMP suite, resolved against $POLYBENCH — which must point at a
# checkout: this repo vendors no kernels of its own.
ALL_KERNELS=(
    "datamining/covariance/covariance-omp.c"
    "datamining/correlation/correlation-omp.c"
    "stencils/jacobi-1d/jacobi-1d-omp.c"
    "stencils/heat-3d/heat-3d-omp.c"
    "stencils/fdtd-2d/fdtd-2d-omp.c"
    "stencils/jacobi-2d/jacobi-2d-omp.c"
    "stencils/adi/adi-omp.c"
    "linear-algebra/blas/gemm/gemm-omp.c"
    "linear-algebra/blas/gesummv/gesummv-omp.c"
    "linear-algebra/blas/trmm/trmm-omp.c"
    "linear-algebra/blas/gemver/gemver-omp.c"
    "linear-algebra/blas/syrk/syrk-omp.c"
    "linear-algebra/blas/syr2k/syr2k-omp.c"
    "linear-algebra/blas/symm/symm-omp.c"
    "linear-algebra/solvers/gramschmidt/gramschmidt-omp.c"
    "linear-algebra/solvers/lu/lu-omp.c"
    "linear-algebra/solvers/cholesky/cholesky-omp.c"
    "linear-algebra/solvers/ludcmp/ludcmp-omp.c"
    "linear-algebra/solvers/trisolv/trisolv-omp.c"
    "linear-algebra/solvers/durbin/durbin-omp.c"
    "linear-algebra/kernels/mvt/mvt-omp.c"
    "linear-algebra/kernels/atax/atax-omp.c"
    "linear-algebra/kernels/doitgen/doitgen-omp.c"
    "linear-algebra/kernels/bicg/bicg-omp.c"
    "linear-algebra/kernels/2mm/2mm-omp.c"
    "linear-algebra/kernels/3mm/3mm-omp.c"
    "medley/floyd-warshall/floyd-warshall-omp.c"
    "medley/deriche/deriche-omp.c"
    "medley/nussinov/nussinov-omp.c"
    "stencils/seidel-2d/seidel-2d-omp.c"
)

# Pick the kernel list: an explicit KERNELS override wins, otherwise the whole
# suite runs. Result lands in the KERNEL_LIST array.
select_kernels() {
    if [ -n "${KERNELS:-}" ]; then
        read -ra KERNEL_LIST <<< "$KERNELS"
    else
        KERNEL_LIST=("${ALL_KERNELS[@]}")
    fi
}

# Resolve a kernel argument: an absolute/relative path, or a path under
# $POLYBENCH. Echoes the resolved path; non-zero if it cannot be found.
resolve_src() {
    local k="$1"
    if [ -f "$k" ]; then echo "$k"; return 0; fi
    if [ -f "$POLYBENCH/$k" ]; then echo "$POLYBENCH/$k"; return 0; fi
    return 1
}
