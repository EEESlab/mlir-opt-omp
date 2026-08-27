/* omp-shape-probe.c — which OpenMP shape our lowering breaks on GAP8
 *
 * barrier-probe.c cleared the runtime: the team barrier takes 512 consecutive
 * calls, ext_pi_cl_nb_cores reports 8 on every core, empty chunks reach the
 * barrier, and seidel-2d's own bounds with our chunk arithmetic run to the last
 * iteration. But that probe is compiled by the SDK's gcc, and seidel-2d only
 * hangs as opt_par — our lowering, with the fork. Same evidence from the
 * driver: ref_seq, ref_par and opt_seq all complete, opt_par does not.
 *
 * So the shape is fine and the runtime is fine; what is left is what we emit
 * for this shape. This probe is written with real pragmas so it goes through
 * ClangIR -> mlir-opt-omp -> PULP_LLC like a kernel does, and walks from the
 * shape every other kernel uses to seidel-2d's, one step at a time:
 *
 *   1  parallel { for }                    the common shape, 29 kernels use it
 *   2  parallel { for; for }               two loops, a barrier between them
 *   3  parallel { for(k) { for } }         a wsloop inside a sequential loop
 *   4  parallel { for(t) { for(w) { for } } }   two levels, seidel's nesting
 *   5  as 4, with bounds computed inside the region   seidel-2d exactly
 *
 * Each stage is its own parallel region and cluster_main prints between them,
 * so the last line printed names the first shape that does not come back. Stage
 * 1 hanging would mean something much more basic than seidel is wrong; stage 5
 * alone hanging points at the runtime-computed bounds rather than the nesting.
 *
 * Run it through the opt_par cell, which is the one that fails:
 *
 *   CELL=opt_par ./run_pulp_probe.sh pulp-probes/omp-shape-probe.c
 *
 * and again with CELL=ref_par to confirm the SDK's own OpenMP takes all five,
 * which is the comparison that makes a stage failure ours.
 */

#include <stdio.h>
#include "pmsis.h"

/* seidel-2d at MINI. */
#define PB_TSTEPS 2
#define PB_N      32

/* Small enough for L1, and static so no allocator is involved: the probe is
   about control flow, not memory. volatile keeps the bodies from folding. */
static volatile float A[PB_N][PB_N];

static void touch(int i) { A[i % PB_N][0] = A[i % PB_N][0] + 1.0f; }

/* --- 1: the shape 29 kernels use -------------------------------------- */
static void stage1(void) {
  int i;
  #pragma omp parallel
  {
    #pragma omp for
    for (i = 0; i < PB_N; i++) touch(i);
  }
}

/* --- 2: two loops, so one real barrier between them --------------------- */
static void stage2(void) {
  int i;
  #pragma omp parallel
  {
    #pragma omp for
    for (i = 0; i < PB_N; i++) touch(i);
    #pragma omp for
    for (i = 0; i < PB_N; i++) touch(i);
  }
}

/* --- 3: one sequential loop around the wsloop --------------------------- */
/* The first shape the barrier-elim pass declines, and the first where the team
   barrier runs more than once per fork. */
static void stage3(void) {
  int i, k;
  #pragma omp parallel private(i, k)
  for (k = 0; k < 16; k++) {
    #pragma omp for
    for (i = 0; i < PB_N; i++) touch(i);
  }
}

/* --- 4: seidel's nesting, but with bounds fixed at compile time ---------- */
static void stage4(void) {
  int i, t, w;
  #pragma omp parallel private(i, t, w)
  for (t = 0; t <= PB_TSTEPS - 1; t++)
    for (w = 3; w <= 3 * (PB_N - 2); w++) {
      #pragma omp for
      for (i = 1; i <= PB_N - 2; i++) touch(i);
    }
}

/* --- 5: seidel-2d exactly ----------------------------------------------- */
/* Same private clause, same skewed bounds computed inside the region, so the
   wsloop's trip count changes on every w and runs as low as one iteration
   across eight cores. Only the body differs from the real kernel. */
static void stage5(void) {
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

void cluster_main(void) {
  printf("=== omp shape probe ===\n");

  printf("[1] parallel { for } ...\n");                       stage1();
  printf("[1] ok\n");

  printf("[2] parallel { for; for } ...\n");                  stage2();
  printf("[2] ok\n");

  printf("[3] parallel { for(k) { for } } ...\n");            stage3();
  printf("[3] ok\n");

  printf("[4] parallel { for(t) { for(w) { for } } } ...\n"); stage4();
  printf("[4] ok\n");

  printf("[5] seidel-2d shape, bounds computed inside ...\n"); stage5();
  printf("[5] ok\n");

  printf("=== all stages returned ===\n");
}
