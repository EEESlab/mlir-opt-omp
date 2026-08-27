/* shape-stages.h — one OpenMP shape per binary, on the way from the shape
 * twenty-nine kernels use to seidel-2d's.
 *
 * omp-shape-probe.c ran all five in one binary and printed between them, which
 * works everywhere except where it is needed: on gvsoc the console output only
 * reaches the log when the program exits, so a run that hangs prints nothing at
 * all — not even the lines that already executed. Under opt_par it produced an
 * empty log, which said only "it hung", not where.
 *
 * So the stage is chosen at compile time and each binary runs exactly one.
 * The signal is whether the run comes back:
 *
 *     stage N ok     printed at the end -> that shape survives
 *     empty log      -> that shape is where it hangs
 *
 * A define cannot be threaded through both pipelines — PULP_POLYBENCH_DEFS
 * reaches the ClangIR front-end but not the harness Makefile, so the ref_*
 * control would be built without it — hence one .c per stage, each two lines
 * over this header.
 *
 *   for n in 1 2 3 4 5; do
 *     CELL=opt_par ./run_pulp_probe.sh pulp-probes/shape-$n.c
 *   done
 *
 * ref_par is the control: the SDK's own OpenMP takes all five, so a stage that
 * hangs under opt_par is ours.
 */

#ifndef PROBE_STAGE
#error "define PROBE_STAGE (1..5) before including shape-stages.h"
#endif

/* No pmsis.h: under opt_par the front-end is ClangIR, which is not given the
   SDK include path. The kernels do not include it either. */
#include <stdio.h>

/* seidel-2d at MINI. */
#define PB_TSTEPS 2
#define PB_N      32

/* Small enough for L1 and static, so no allocator is involved: these probes are
   about control flow. volatile keeps the bodies from folding away. */
static volatile float A[PB_N][PB_N];

static void touch(int i) { A[i % PB_N][0] = A[i % PB_N][0] + 1.0f; }

#if PROBE_STAGE == 1
/* The shape twenty-nine kernels use: one wsloop, last in its region. */
#define STAGE_NAME "parallel { for }"
static void stage(void) {
  int i;
  #pragma omp parallel
  {
    #pragma omp for
    for (i = 0; i < PB_N; i++) touch(i);
  }
}

#elif PROBE_STAGE == 2
/* Two loops, so one real barrier between them. */
#define STAGE_NAME "parallel { for; for }"
static void stage(void) {
  int i;
  #pragma omp parallel
  {
    #pragma omp for
    for (i = 0; i < PB_N; i++) touch(i);
    #pragma omp for
    for (i = 0; i < PB_N; i++) touch(i);
  }
}

#elif PROBE_STAGE == 3
/* A wsloop inside a sequential loop: the first shape the barrier-elim pass
   declines, and the first where the team barrier runs more than once per fork. */
#define STAGE_NAME "parallel { for(k) { for } }"
static void stage(void) {
  int i, k;
  #pragma omp parallel private(i, k)
  for (k = 0; k < 16; k++) {
    #pragma omp for
    for (i = 0; i < PB_N; i++) touch(i);
  }
}

#elif PROBE_STAGE == 4
/* seidel's nesting, but with the inner bounds fixed at compile time. */
#define STAGE_NAME "parallel { for(t) { for(w) { for } } }"
static void stage(void) {
  int i, t, w;
  #pragma omp parallel private(i, t, w)
  for (t = 0; t <= PB_TSTEPS - 1; t++)
    for (w = 3; w <= 3 * (PB_N - 2); w++) {
      #pragma omp for
      for (i = 1; i <= PB_N - 2; i++) touch(i);
    }
}

#elif PROBE_STAGE == 5
/* seidel-2d exactly: same private clause, same skewed bounds computed inside
   the region, so the trip count changes on every w and falls as low as one
   iteration across eight cores. Only the body differs from the real kernel. */
#define STAGE_NAME "seidel-2d shape, bounds computed inside"
static void stage(void) {
  int i, t, w, ilo, ihi;
  #pragma omp parallel private(i, t, w, ilo, ihi)
  for (t = 0; t <= PB_TSTEPS - 1; t++)
    for (w = 3; w <= 3 * (PB_N - 2); w++) {
      ihi = (w - 1) / 2;
      if (ihi > PB_N - 2) ihi = PB_N - 2;
      ilo = (w - (PB_N - 2) + 1) / 2;
      if (ilo < 1) ilo = 1;

      #pragma omp for
      for (i = ilo; i <= ihi; i++) touch(i);
    }
}

#else
#error "PROBE_STAGE must be 1..5"
#endif

void cluster_main(void) {
  /* Printed before as well as after: on a platform that flushes at exit both
     lines appear together, and on one that does not the first still says the
     binary started. */
  printf("stage %d: %s ...\n", PROBE_STAGE, STAGE_NAME);
  stage();
  printf("stage %d ok\n", PROBE_STAGE);
}
