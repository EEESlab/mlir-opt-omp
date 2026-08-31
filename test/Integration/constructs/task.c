/* task — the task body runs, and its write reaches the caller.
 *
 * The pointer indirection (int *p = &x; *p = 42) is deliberate and is repeated
 * in every task test here: this tool treats a scalar alloca whose first
 * in-region use is a store as a per-thread PRIVATE capture (loop-induction
 * semantics), which would hide a direct `x = 42`. A pointer capture — first
 * use is a load — is packed by value, so the store reaches the caller's x.
 *
 * The parallel region's implicit barrier completes outstanding tasks, so the
 * read after it is ordered without needing a taskwait.
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
