/* wsloop nowait — the loop still computes correctly without its barrier.
 *
 * NOT a test that the barrier is gone: the absence of a barrier is only
 * observable through a data race, and a test that races is nondeterministic —
 * worse than no test, because it fails at random and gets ignored. What this
 * catches is the opposite failure, and the likelier one: dropping the barrier
 * corrupting the loop that precedes it.
 *
 * That the barrier call really is not emitted is asserted at IR level:
 * Regression/wsloop/nowait/nowait-{iomp,libgomp,pmsis}.mlir.
 *
 * The second loop reads what the first wrote, so it is placed after an
 * explicit barrier — with nowait on the first loop there is nothing else
 * keeping the two apart.
 */
#include <stdio.h>

#define N 512
int a[N], b[N];

int main(void) {
  int i;

  #pragma omp parallel num_threads(4) private(i)
  {
    #pragma omp for nowait
    for (i = 0; i < N; i++)
      a[i] = i + 1;

    #pragma omp barrier

    #pragma omp for
    for (i = 0; i < N; i++)
      b[i] = a[i] * 2;
  }

  int bad = 0;
  for (i = 0; i < N; i++)
    if (a[i] != i + 1 || b[i] != 2 * (i + 1)) bad++;
  printf("%d\n", bad == 0 ? 42 : bad);
  return 0;
}
