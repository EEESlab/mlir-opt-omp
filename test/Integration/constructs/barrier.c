/* barrier — an explicit barrier orders what is written before it against what
 * is read after it.
 *
 * Each thread writes its own slot, then every thread sums all of them. The sum
 * is only complete if the barrier held: without it a thread can reach the sum
 * before its neighbours have written, and reads a short total. PolyBench never
 * writes `#pragma omp barrier` — its only barriers are the implicit ones that
 * close a work-sharing loop — so this construct has no coverage there at all.
 *
 * The check is per thread, into a slot of its own, so a failure names how many
 * threads saw an incomplete state rather than just that one did.
 */
#include <stdio.h>
#include <omp.h>

#define T 4
#define WANT (T * (T + 1) / 2)

int data[T];
int bad[256];

int main(void) {
  #pragma omp parallel num_threads(T)
  {
    int t = omp_get_thread_num();
    data[t] = t + 1;

    #pragma omp barrier

    int k, sum = 0;
    for (k = 0; k < T; k++) sum += data[k];
    if (sum != WANT) bad[t] = 1;
  }

  int i, n = 0;
  for (i = 0; i < 256; i++) n += bad[i];
  printf("%d\n", n == 0 ? 42 : n);
  return 0;
}
