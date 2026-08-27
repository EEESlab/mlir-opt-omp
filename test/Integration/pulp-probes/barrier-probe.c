/* barrier-probe.c — why seidel-2d hangs on GAP8, narrowed to one line of output
 *
 * seidel-2d is the only kernel in the suite where a sequential loop sits
 * between `#pragma omp parallel` and `#pragma omp for`, so it is the only one
 * that calls ext_pi_cl_team_barrier repeatedly inside a loop — 176 times at
 * MINI (TSTEPS=2, N=32, w in [3, 90]). Everywhere else the team barrier fires
 * once per region entry. That is the one thing that makes it different, so it
 * is the first thing to rule in or out.
 *
 * Five probes, in order of how much they assume. Each prints before it runs and
 * again after, so the console log localises a hang to one probe: the last line
 * printed is the probe that did not come back.
 *
 *   1  team shape        does ext_pi_cl_nb_cores() report the size we forked?
 *   2  back-to-back      N team barriers in a row, N = 1 .. 512
 *   3  divergent work    unequal work per core between barriers
 *   4  empty chunks      trip count below the team size, then a barrier
 *   5  seidel shape      the real bounds, and the chunk arithmetic our
 *                        lowering emits, with nothing of our lowering in it
 *
 * Probe 5 is the decisive one. If 1-4 pass and 5 hangs, the arithmetic is
 * wrong for seidel's bounds. If 2 hangs, the barrier shim cannot take
 * consecutive calls and the kernel is incidental.
 *
 * Nothing here goes through mlir-opt-omp: the probe is plain C against the
 * same ext_pi_* shims (utilities/interface-adapter.c) the generated code calls,
 * so a failure here is the runtime's, not the compiler's.
 *
 * Build and run inside the PolyBench-PULP harness, as a kernel:
 *
 *   cd $PULP_APP_DIR
 *   make clean all run KERNEL_SRC=<repo>/test/Integration/pulp-probes/barrier-probe.c
 *
 * No OMP_NATIVE and no OMP_OPT: the probe *is* the kernel, and it wants
 * neither the gcc OpenMP runtime nor a kernel.o.
 */

#include <stdio.h>
#include "pmsis.h"

/* The shims the generated code calls, from utilities/interface-adapter.c. */
extern void ext_pi_cl_team_fork(int nb_cores, void (*entry)(void *), void *arg);
extern void ext_pi_cl_team_barrier(void);
extern int  ext_pi_core_id(void);
extern int  ext_pi_cl_nb_cores(void);

/* What rules.dsl hands ext_pi_cl_team_fork when no num_threads clause asks for
   something else (pmsis default_team_size). */
#define TEAM 8

/* seidel-2d at MINI. */
#define PB_TSTEPS 2
#define PB_N      32

/* Written from inside the team, read by core 0 after the fork returns. */
static volatile int seen_nb[TEAM];
static volatile int seen_id[TEAM];
static volatile int probe_arg;      /* how many barriers probe 2 should do */
static volatile int reached_w;      /* last w probe 5 got through */
static volatile int reached_t;

/* --- 1: team shape ------------------------------------------------------- */
/* The chunk formula divides by ext_pi_cl_nb_cores(). If that returns 0 the
   division does not trap on RISC-V, it yields -1, and every core computes an
   empty chunk — wrong results rather than a hang, but worth excluding first. */
static void probe_team(void *arg) {
  int id = ext_pi_core_id();
  int nb = ext_pi_cl_nb_cores();
  if (id >= 0 && id < TEAM) { seen_id[id] = id; seen_nb[id] = nb; }
}

/* --- 2: consecutive barriers --------------------------------------------- */
/* The hypothesis this probe exists for: a hardware barrier that needs re-arming
   between uses, or a counter not reset, would survive one call and deadlock on
   a run of them. 176 is what seidel-2d asks for at MINI. */
static void probe_barriers(void *arg) {
  int n = probe_arg;
  for (int k = 0; k < n; k++)
    ext_pi_cl_team_barrier();
}

/* --- 3: divergent work between barriers ---------------------------------- */
/* Cores arrive at the barrier well apart in time. A barrier that assumes the
   cores are roughly in step fails here and nowhere else. */
static void probe_divergent(void *arg) {
  int id = ext_pi_core_id();
  for (int k = 0; k < 64; k++) {
    volatile int sink = 0;
    for (int i = 0; i < id * 50; i++) sink++;
    ext_pi_cl_team_barrier();
  }
}

/* --- 4: chunks smaller than the team ------------------------------------- */
/* seidel's inner loop has as few as one iteration for eight cores, so seven
   cores run no iterations at all and fall straight through to the barrier.
   That path has to reach the barrier too. */
static void probe_empty_chunks(void *arg) {
  int id = ext_pi_core_id();
  int nc = ext_pi_cl_nb_cores();
  if (nc <= 0) return;                    /* probe 1 already reported this */

  for (int trip = 1; trip <= TEAM + 2; trip++) {
    int lb = 0, ub = trip, step = 1;
    int chunk = ((ub - lb + step - 1) / step + nc - 1) / nc;
    int lo = lb + id * chunk * step;
    int hi = lo + chunk * step;
    if (hi > ub) hi = ub;

    volatile int sink = 0;
    for (int i = lo; i < hi; i += step) sink++;
    ext_pi_cl_team_barrier();
  }
}

/* --- 5: the seidel-2d shape ---------------------------------------------- */
/* The bounds are seidel's own, and the four lines computing lo/hi are the block
   chunking OmpOutliningPass emits for `emit thread_bounds`:
 *
 *     trip  = ceil((ub - lb) / step)
 *     chunk = ceil(trip / num_threads)
 *     lo    = lb + id * chunk * step
 *     hi    = min(lo + chunk * step, ub)
 *
 * transcribed rather than generated, so this probe fails only if the arithmetic
 * itself is wrong on this hardware. reached_t/reached_w say how far it got. */
static void probe_seidel_shape(void *arg) {
  int id = ext_pi_core_id();
  int nc = ext_pi_cl_nb_cores();
  if (nc <= 0) return;

  for (int t = 0; t <= PB_TSTEPS - 1; t++)
    for (int w = 3; w <= 3 * (PB_N - 2); w++) {
      int ihi = (w - 1) / 2;
      if (ihi > PB_N - 2) ihi = PB_N - 2;
      int ilo = (w - (PB_N - 2) + 1) / 2;
      if (ilo < 1) ilo = 1;

      int lb = ilo, ub = ihi + 1, step = 1;      /* omp.wsloop is [lb, ub) */
      int trip  = (ub - lb + step - 1) / step;
      int chunk = (trip + nc - 1) / nc;
      int lo    = lb + id * chunk * step;
      int hi    = lo + chunk * step;
      if (hi > ub) hi = ub;

      volatile int sink = 0;
      for (int i = lo; i < hi; i += step) sink++;

      /* Only core 0 writes, so this cannot itself be the divergence. */
      if (id == 0) { reached_t = t; reached_w = w; }
      ext_pi_cl_team_barrier();
    }
}

/* ------------------------------------------------------------------------- */

void cluster_main(void) {
  printf("=== barrier probe, team %d ===\n", TEAM);

  /* 1 */
  printf("[1] team shape ...\n");
  for (int i = 0; i < TEAM; i++) { seen_id[i] = -1; seen_nb[i] = -1; }
  ext_pi_cl_team_fork(TEAM, probe_team, 0);
  printf("[1] done. core_id/nb_cores seen:");
  for (int i = 0; i < TEAM; i++) printf(" %d/%d", seen_id[i], seen_nb[i]);
  printf("\n");
  if (seen_nb[0] != TEAM)
    printf("[1] MISMATCH: forked %d, nb_cores reports %d — the chunk formula "
           "divides by this\n", TEAM, seen_nb[0]);

  /* 2 */
  static const int runs[] = { 1, 2, 8, 64, 176, 512 };
  for (unsigned r = 0; r < sizeof runs / sizeof runs[0]; r++) {
    printf("[2] %d consecutive barriers ...\n", runs[r]);
    probe_arg = runs[r];
    ext_pi_cl_team_fork(TEAM, probe_barriers, 0);
    printf("[2] %d ok\n", runs[r]);
  }

  /* 3 */
  printf("[3] divergent work between 64 barriers ...\n");
  ext_pi_cl_team_fork(TEAM, probe_divergent, 0);
  printf("[3] ok\n");

  /* 4 */
  printf("[4] trip counts 1..%d over %d cores ...\n", TEAM + 2, TEAM);
  ext_pi_cl_team_fork(TEAM, probe_empty_chunks, 0);
  printf("[4] ok\n");

  /* 5 */
  printf("[5] seidel-2d shape, %d barriers ...\n",
         PB_TSTEPS * (3 * (PB_N - 2) - 3 + 1));
  reached_t = -1; reached_w = -1;
  ext_pi_cl_team_fork(TEAM, probe_seidel_shape, 0);
  printf("[5] ok, reached t=%d w=%d\n", reached_t, reached_w);

  printf("=== all probes returned ===\n");
}
