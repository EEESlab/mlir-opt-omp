/* Driver for vecadd.c — fills two vectors, calls the (ClangIR-lowered) kernel,
 * and prints the result so the run is self-checking.
 *
 * Built with the STOCK clang (not through ClangIR) and linked against the
 * kernel object: only the OpenMP kernel needs the CIR / mlir-opt-omp path, so
 * the driver stays out of it — the same split quick-compile/ uses.
 *
 * Expected output for N = 8 (c[i] = a[i] + b[i] = i + 10*i = 11*i):
 *   0 11 22 33 44 55 66 77
 */
#include <stdio.h>

void vecadd(int n, const int *a, const int *b, int *c);

int main(void) {
  enum { N = 8 };
  int a[N], b[N], c[N] = {0};
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = 10 * i;
  }

  vecadd(N, a, b, c);

  for (int i = 0; i < N; ++i)
    printf("%d\n", c[i]);
  return 0;
}
