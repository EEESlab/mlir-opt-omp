/* taskwait — the wait orders the task's write against the read that follows.
 *
 * Detecting a MISSING taskwait is probabilistic, and the test says so rather
 * than pretending otherwise: without the wait the read is merely *allowed* to
 * observe the pre-task value, and a one-statement task usually completes
 * before the reader gets there anyway. The task therefore does enough work to
 * hold the window open — without it, removing the taskwait still printed 42.
 *
 * That the runtime's wait call is actually emitted is asserted deterministically
 * at IR level instead: Regression/taskwait/taskwait-{iomp,libgomp}.mlir.
 *
 * See task.c for why the write goes through a pointer.
 */
#include <stdio.h>

#define WORK 200000

int main(void) {
  int x = 0, y = 0, acc = 0;
  int *px = &x, *py = &y, *pacc = &acc;

  #pragma omp parallel num_threads(4)
  {
    #pragma omp task
    {
      int i, s = 0;
      for (i = 0; i < WORK; i++) s += i % 7;   /* hold the window open */
      *pacc = s;
      *px = 42;
    }
    #pragma omp taskwait
    *py = *px;          /* ordered after the task: reads 42 */
  }

  printf("%d\n", y);
  return 0;
}
