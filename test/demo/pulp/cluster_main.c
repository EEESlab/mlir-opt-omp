/* Cluster-side driver for ../vecadd.c — the PMSIS counterpart of ../driver.c.
 *
 * Runs on cluster core 0 (dispatched by pulp_main.c). It calls the SAME kernel
 * as the host demo, but that kernel now lives in kernel.o, lowered with
 * --omp-lower-runtime=pmsis: its `#pragma omp parallel for` became an
 * ext_pi_cl_team_fork over the 8 cluster cores + a work-shared loop. Core 0
 * then prints the result to the gvsoc console.
 *
 * Expected output (same as the host demo, c[i] = i + 10*i = 11*i):
 *   0 11 22 33 44 55 66 77
 */
#include <stdio.h>

void vecadd(int n, const int *a, const int *b, int *c);

void cluster_main() {
  enum { N = 8 };
  int a[N], b[N], c[N] = {0};
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = 10 * i;
  }

  vecadd(N, a, b, c);

  for (int i = 0; i < N; ++i)
    printf("%d\n", c[i]);
}
