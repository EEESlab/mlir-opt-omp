/*#include <stdio.h>

void mul(int a[3][3], int b[3][3], int c[3][3], int m, int n, int k);

int main() {
  int a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
  int b[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
  int c[3][3] = {{1,1,1},{1,1,1},{1,1,1}};

  mul(a,b,c,3,3,3);

  for (int i = 0; i < 3; ++i)
   for(int j = 0; j < 3; ++j)
    printf("%d\n", c[i][j]);

  return 0;
}
*/

#include <stdio.h>
#include <stdlib.h>

#define _PB_NI 128
#define _PB_NJ 128
#define _PB_NK 128

// Dichiarazione esterna del kernel
extern void gemm(double alpha, double beta, double A[_PB_NI][_PB_NK], double B[_PB_NK][_PB_NJ], double C[_PB_NI][_PB_NJ]);

int main() {
    double alpha = 1.5;
    double beta = 1.2;

    // Allocazione delle matrici
    double (*A)[_PB_NK] = malloc(sizeof(double) * _PB_NI * _PB_NK);
    double (*B)[_PB_NJ] = malloc(sizeof(double) * _PB_NK * _PB_NJ);
    double (*C)[_PB_NJ] = malloc(sizeof(double) * _PB_NI * _PB_NJ);

    // Inizializzazione dati
    for (int i = 0; i < _PB_NI; i++) {
        for (int k = 0; k < _PB_NK; k++) A[i][k] = (double)(i * k % 100) / 100.0;
        for (int j = 0; j < _PB_NJ; j++) C[i][j] = (double)(i * j % 100) / 100.0;
    }
    for (int k = 0; k < _PB_NK; k++) {
        for (int j = 0; j < _PB_NJ; j++) B[k][j] = (double)(k * j % 100) / 100.0;
    }

    // Esecuzione del kernel OpenMP
    gemm(alpha, beta, A, B, C);

    // Stampa di verifica (impedisce che il compilatore rimuova il kernel per ottimizzazione dead-code)
    printf("Risultato in C[0][0]: %f\n", C[0][0]);
    printf("Risultato in C[%d][%d]: %f\n", _PB_NI-1, _PB_NJ-1, C[_PB_NI-1][_PB_NJ-1]);

    // Cleanup
    free(A);
    free(B);
    free(C);

    return 0;
}
