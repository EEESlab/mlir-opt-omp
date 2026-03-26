/*#include <stdio.h>
#include <omp.h>

void mul(int a[3][3], int b[3][3], int c[3][3], int m, int n, int o) {
  int i, j, k;
  int alpha = 7;
  int beta = 4;
  #pragma omp parallel private (j, k)
  {
    #pragma omp for
    for (i = 0; i < m; i++)
    {
      // DEBUG i
      if (i < 5)
        printf("[DBG] i = %d\n", i);

      for (j = 0; j < n; j++)
      {

        // DEBUG j
        if (i == 0 && j < n)
          printf("[DBG]   j = %d\n", j);

        c[i][j] *= beta;

        for (k = 0; k < o; ++k) {
          c[i][j] += alpha * a[i][k] * b[k][j];

        }
      }
    }
  }
}
*/
#include <omp.h>
#include <stdio.h>

#define _PB_NI 128
#define _PB_NJ 128
#define _PB_NK 128

void gemm(double alpha, double beta, double A[_PB_NI][_PB_NK], double B[_PB_NK][_PB_NJ], double C[_PB_NI][_PB_NJ]) {
  int i=42, j, k;

  #pragma omp parallel private (j, k)
  {
    printf("THREADS = %d\n", omp_get_num_threads());
    /* C := alpha*A*B + beta*C */
    #pragma omp for
    for (i = 0; i < _PB_NI; i++) {
      for (j = 0; j < _PB_NJ; j++) {
        C[i][j] *= beta;
        for (k = 0; k < _PB_NK; ++k) {
          C[i][j] += alpha * A[i][k] * B[k][j];
        }
      }
    }
  }
}
