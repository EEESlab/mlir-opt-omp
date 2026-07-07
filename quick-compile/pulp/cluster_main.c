/* Cluster-side driver for ../test.c — the PULP counterpart of ../main.c.
 * Runs on cluster core 0 (dispatched by pulp_main.c), calls the kernel from
 * ../test.o and prints the result to the gvsoc console. */
#include <stdio.h>

void add(int a[], int b[], int c[], int n);

void cluster_main() {
  int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int b[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  int c[10] = {0};

  add(a, b, c, 10);

  for (int i = 0; i < 10; ++i)
    printf("%d\n", c[i]);
}
