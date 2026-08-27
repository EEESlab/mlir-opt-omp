/* jump-threading.c — the smallest program that shows why gcc is counted at -O0.
 *
 * Four work-sharing loops in one parallel region. gcc elides the implicit
 * barrier of the last one, so three team barriers are needed and three is what
 * -O0 emits.
 *
 * From -O1 on, gcc emits more call sites than that — six here — without adding
 * a single barrier to any execution. Jump threading duplicates the blocks the
 * calls sit in: the path where a thread's chunk is empty is split off and
 * carries its own copy of the barrier sequence. `-fno-thread-jumps` brings the
 * count back to three at every level, which is the whole argument.
 *
 * Deliberately free of PolyBench: one file, one compiler, no headers, so the
 * count can be reproduced with a single command.
 *
 *   gcc -fopenmp -O0 -S jump-threading.c -o - | grep -c GOMP_barrier   # 3
 *   gcc -fopenmp -O3 -S jump-threading.c -o - | grep -c GOMP_barrier   # 6
 *   gcc -fopenmp -O3 -fno-thread-jumps -S jump-threading.c -o - \
 *       | grep -c GOMP_barrier                                         # 3
 *
 * Count GOMP_barrier and not `call GOMP_barrier`: at -O2 and above gcc turns
 * the last one into a tail call, which prints as `jmp` and which a regex
 * anchored on `call` silently drops.
 */

void kernel(int n, double *a, double *b, double *c, double *d) {
  int i, j;
  #pragma omp parallel private(j)
  {
    /* A nested loop, so the body is big enough that gcc keeps the loop rather
       than folding it away and taking the barrier with it. */
    #pragma omp for
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++)
        a[i] += b[j] * c[j];

    #pragma omp for
    for (i = 0; i < n; i++) b[i] = a[i] * 2.0;

    #pragma omp for
    for (i = 0; i < n; i++) c[i] = b[i] + a[i];

    /* Last in the region: gcc elides this one's barrier, at every level. */
    #pragma omp for
    for (i = 0; i < n; i++) d[i] = c[i] - b[i];
  }
}
