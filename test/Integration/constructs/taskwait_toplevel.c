/* task + taskwait OUTSIDE any parallel region.
 *
 * A distinct path through the lowering, not a variation on taskwait.c: with no
 * enclosing region there is no outlined function to take a thread id from, so
 * the plan pass has to resolve a real global_tid on its own (a memoised
 * __kmpc_global_thread_num on iomp). That path previously emitted an undefined
 * gtid and crashed, which is why it has a test of its own.
 *
 * There is also no implicit barrier out here to complete the task, so the
 * taskwait is the only thing ordering the write against the read.
 */
#include <stdio.h>

int main(void) {
  int x = 0;
  int *p = &x;

  #pragma omp task
  {
    *p = 42;
  }
  #pragma omp taskwait

  printf("%d\n", x);
  return 0;
}
