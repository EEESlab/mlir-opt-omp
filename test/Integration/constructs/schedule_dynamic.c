/* wsloop schedule(dynamic) — every iteration runs exactly once.
 *
 * The counter array is the whole test: a distribution that loses iterations
 * leaves zeros, one that hands the same chunk out twice leaves twos, and both
 * are reported by name rather than as a bare mismatch. This is the property a
 * chunked schedule has to preserve and the one a wrong bound computation
 * breaks first — an off-by-one on the inclusive upper bound shows up here as a
 * single missed iteration per chunk.
 *
 * N is not a multiple of the team size on purpose, so the last chunk is short.
 *
 * If the REFERENCE build of this one fails to link with an undefined
 * `__kmpc_dispatch_deinit`, that is the environment and not the test: the
 * ClangIR fork emits a newer libomp API than an older system libomp provides.
 * Our own lowering does not emit that call at all, so the opt build links
 * either way — which is why the driver reports the two sides separately
 * instead of only their agreement.
 */
#include <stdio.h>

#define N 1001
int hits[N];

int main(void) {
  int i;

  #pragma omp parallel num_threads(4) private(i)
  {
    #pragma omp for schedule(dynamic)
    for (i = 0; i < N; i++)
      hits[i] = hits[i] + 1;
  }

  int missed = 0, twice = 0;
  for (i = 0; i < N; i++) {
    if (hits[i] == 0) missed++;
    else if (hits[i] > 1) twice++;
  }
  if (missed == 0 && twice == 0) printf("42\n");
  else printf("missed=%d twice=%d\n", missed, twice);
  return 0;
}
