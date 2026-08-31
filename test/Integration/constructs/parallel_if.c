/* parallel if(0) — a false condition runs the region serialized.
 *
 * Counted the same way num_threads.c counts: exactly one slot may be marked.
 * An ignored `if` leaves the default team size, so the count comes out as the
 * core count rather than 1. num_threads(4) is on the directive as well, so a
 * pass cannot come from the machine happening to be single-core.
 */
#include <stdio.h>
#include <omp.h>

int seen[256];

int main(void) {
  #pragma omp parallel num_threads(4) if(0)
  {
    seen[omp_get_thread_num()] = 1;
  }

  int i, n = 0;
  for (i = 0; i < 256; i++) n += seen[i];
  printf("%d\n", n == 1 ? 42 : n);
  return 0;
}
