/* End-to-end C smoke test for omp.taskwait lowering.
 *
 * A task writes 42 through a shared pointer; the value is read back later in
 * the same parallel region, AFTER an explicit `#pragma omp taskwait`, and
 * copied into the printed output.  The taskwait is load-bearing: without it the
 * read may observe the pre-task value (0) because the runtime may defer the
 * task.  With it, the read — and the printed result — is deterministically 42.
 *
 * The pointer indirection (int *px = &x; *px = 42) is deliberate, matching
 * task_smoke.c: a scalar alloca whose first in-region use is a store is treated
 * as a per-thread private capture, which would hide a direct write; a pointer
 * capture (first use is a load) is packed by value, so the store reaches the
 * caller's x.  Every thread writes the same value (42), so the output stays
 * deterministic under num_threads > 1.
 *
 * Compiled both with the stock compiler (gcc -fopenmp) and through the CIR /
 * mlir-opt-omp pipeline; the two outputs must match (42).  Depends on ClangIR
 * emitting omp.taskwait; see run_tasks.sh (check [4]).
 */
#include <stdio.h>

int main(void) {
  int x = 0, y = 0;
  int *px = &x, *py = &y;

  #pragma omp parallel num_threads(4)
  {
    #pragma omp task
    {
      *px = 42;
    }
    #pragma omp taskwait
    *py = *px;            /* runs only after the task completed: reads 42 */
  }

  printf("%d\n", y);
  return 0;
}
