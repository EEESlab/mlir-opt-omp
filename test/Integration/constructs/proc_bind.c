/* parallel proc_bind — accepted and harmless.
 *
 * NOT a test that the binding took effect: where a thread ends up is not
 * observable portably from inside the program, and a test that pretends
 * otherwise would be measuring the machine. What this pins down is that the
 * clause reaches the lowering and the region still runs with the right team
 * afterwards — the failure it would catch is a clause that breaks the code
 * around it.
 *
 * That the policy is actually handed to the runtime call is asserted at IR
 * level instead: Regression/parallel/proc_bind/proc_bind-{iomp,libgomp}.mlir,
 * and proc_bind-pmsis.mlir for the diagnostic where it is unsupported.
 */
#include <stdio.h>
#include <omp.h>

#define T 4
int seen[256];

int main(void) {
  #pragma omp parallel num_threads(T) proc_bind(close)
  {
    seen[omp_get_thread_num()] = 1;
  }

  int i, n = 0;
  for (i = 0; i < 256; i++) n += seen[i];
  printf("%d\n", n == T ? 42 : n);
  return 0;
}
