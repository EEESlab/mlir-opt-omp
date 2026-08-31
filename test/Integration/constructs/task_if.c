/* task if(0) — a false condition runs the task UNDEFERRED, on the spot.
 *
 * The read-back sits immediately after the construct with NO taskwait between
 * them, which is the whole point: if the task were deferred the read would be
 * free to observe the pre-task 0, and only the region's implicit barrier would
 * fix it up later. Reading 42 here means the body already ran.
 *
 * One thread, so the result cannot depend on which thread got there first.
 * iomp takes the __kmpc_omp_task_begin_if0 / direct call / _complete_if0
 * branch; libgomp passes if_clause=false to GOMP_task, which runs it inline.
 */
#include <stdio.h>

int main(void) {
  int x = 0, y = 0;
  int *px = &x, *py = &y;

  #pragma omp parallel num_threads(1)
  {
    #pragma omp task if(0)
    {
      *px = 42;
    }
    *py = *px;          /* undeferred: already 42, with no taskwait */
  }

  printf("%d\n", y);
  return 0;
}
