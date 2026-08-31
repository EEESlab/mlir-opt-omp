/* parallel firstprivate — copied in at entry, and not copied back out.
 *
 * Both halves are checked, because each fails differently:
 *   copy-in    every thread must observe the initial 42 in its own copy
 *   privacy    the increment each thread performs must NOT reach the original
 *
 * Testing only the first would pass even if the clause were ignored entirely:
 * a shared x also reads 42 at entry. It is the write that separates them — on
 * a shared x the increments race and the value outside is no longer 42.
 */
#include <stdio.h>
#include <omp.h>

#define T 4
int saw[T];

int main(void) {
  int x = 42;

  #pragma omp parallel num_threads(T) firstprivate(x)
  {
    saw[omp_get_thread_num()] = x;   /* copy-in: 42 in every thread */
    x += 100;                        /* private: must not escape */
  }

  int i, ok = (x == 42);
  for (i = 0; i < T; i++) if (saw[i] != 42) ok = 0;
  printf("%d\n", ok ? 42 : x);
  return 0;
}
