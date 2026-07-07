/* PMSIS interface adapter — resolves the ext_pi_* shims that kernels lowered
 * with --omp-lower-runtime=pmsis call into the real PMSIS API. Copy of the
 * canonical reference in docs/pmsis-interface-adapter.c; keep the two in
 * sync. */

#include <pmsis.h>

void ext_pi_cl_team_barrier()
{
  pi_cl_team_barrier();
}

int ext_pi_core_id()
{
  return pi_core_id();
}

int omp_get_thread_num()
{
  return pi_core_id();
}

void ext_pi_cl_team_fork(int nb_cores, void (*entry)(void *), void *arg)
{
  pi_cl_team_fork(nb_cores, entry, arg);
}

int ext_pi_cl_nb_cores()
{
  return pi_cl_team_nb_cores();
}
