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
