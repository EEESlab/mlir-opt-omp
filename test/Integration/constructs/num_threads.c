/* parallel num_threads — the team really has the size that was asked for.
 *
 * Each thread marks its own slot, and the marks are counted after the region.
 * A clause that was parsed and dropped leaves the runtime's default team size,
 * which on any machine with other than WANT cores prints that number instead
 * of 42. WANT is deliberately not a power of two and larger than a typical
 * core count, so "the default happened to match" is not a way to pass.
 */
#include <stdio.h>
#include <omp.h>

#define WANT 6
int seen[256];

int main(void) {
  #pragma omp parallel num_threads(WANT)
  {
    seen[omp_get_thread_num()] = 1;
  }

  int i, n = 0;
  for (i = 0; i < 256; i++) n += seen[i];
  printf("%d\n", n == WANT ? 42 : n);
  return 0;
}
