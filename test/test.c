void add(int a[], int b[], int c[], int n) {
  int i;
  #pragma omp parallel for
  for( i=0; i<n; i++)
    c[i] = a[i] + b[i];
}
