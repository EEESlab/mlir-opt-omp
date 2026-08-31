/* task firstprivate — copied in at creation, and writes to the copy stay in it.
 *
 * What this pins down is the SEMANTICS, not the clause in isolation, and the
 * difference is worth stating because it was measured: removing
 * `firstprivate(x)` from this file changes nothing on either runtime. That is
 * not the test being weak, it is a property of the lowering — a capture whose
 * first in-region use is a load is packed by value anyway, so the default
 * already gives copy-in and non-escaping writes for this shape. The clause is
 * observationally a no-op here.
 *
 * So: this file is evidence that a task with firstprivate behaves correctly
 * end to end. Evidence that the clause is READ and lowered is a different
 * thing, and lives at IR level in
 * Regression/task/firstprivate/task-firstprivate-{iomp,libgomp}.mlir, where a
 * missing clause changes the emitted code and the CHECK lines fail.
 *
 * The write is checked rather than the read, because only the write is
 * deterministic: the obvious test — change x after creating the task, see
 * which value it picked up — depends on the task running late enough for the
 * change to matter, and a one-statement task does not.
 */
#include <stdio.h>

int main(void) {
  int x = 42, seen = 0;
  int *ps = &seen;

  #pragma omp parallel num_threads(1)
  {
    #pragma omp task firstprivate(x)
    {
      *ps = x;          /* the copy taken at creation: 42 */
      x = 999;          /* into the copy — must not escape */
    }
    #pragma omp taskwait
  }

  printf("%d\n", (x == 42 && seen == 42) ? 42 : x);
  return 0;
}
