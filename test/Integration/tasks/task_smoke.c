/* End-to-end C smoke test for omp.task lowering.
 *
 * A task nested in a parallel region writes 42 through a shared pointer.  The
 * pointer indirection (int *p = &x; *p = 42) is deliberate: this tool treats a
 * scalar alloca whose first in-region use is a store as a per-thread *private*
 * capture (loop-IV semantics), which would hide a direct `x = 42`.  A pointer
 * capture (first use is a load) is packed by value, so the store reaches the
 * caller's x.
 *
 * After the parallel region's implicit barrier (which completes outstanding
 * tasks) main prints 42.  Compiled both with the stock compiler (gcc -fopenmp)
 * and through the CIR / mlir-opt-omp pipeline; the two outputs must match (42).
 *
 * No omp_* runtime calls and no PolyBench support code, so it builds
 * standalone — see run_tasks.sh.
 */
#include <stdio.h>

int main(void) {
  int x = 0;
  int *p = &x;

  #pragma omp parallel num_threads(4)
  {
    #pragma omp task
    {
      *p = 42;
    }
  }

  printf("%d\n", x);
  return 0;
}
