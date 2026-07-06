/* Reference implementation of the PMSIS interface adapter.
 *
 * The pmsis runtime in rules.dsl lowers OpenMP constructs to calls into a
 * small shim layer (`ext_pi_*`) rather than into the PMSIS API directly, so
 * the generated code stays independent of the PMSIS headers. Any harness that
 * links a kernel produced with --omp-lower-runtime=pmsis must provide these
 * symbols; the PolyBench-PULP harness used by test/Integration (PULP_APP_DIR)
 * ships its own copy. This file is kept as the canonical reference — it is
 * not built by this repo.
 */

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
