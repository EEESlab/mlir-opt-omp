/* Driver for test.c — prints the result so ./test and ./test-ref can be
 * compared by eye or diffed. */
#include <stdio.h>

void add(int a[], int b[], int c[], int n);

int main(void) {
  int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int b[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  int c[10] = {0};

  add(a, b, c, 10);

  for (int i = 0; i < 10; ++i)
    printf("%d\n", c[i]);

  return 0;
}
