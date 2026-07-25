/* PMSIS interface adapter — resolves the ext_pi_* shims that a kernel lowered
 * with --omp-lower-runtime=pmsis calls, mapping them onto the real PMSIS API.
 *
 * This is the layer that makes the DSL's runtime-agnostic `ext_pi_*` calls
 * concrete on GAP8: the team fork, the barrier, and the core-id / core-count
 * queries the work-sharing loop uses for its per-core bounds. Kept in sync with
 * quick-compile/pulp/interface-adapter.c.
 */
#include <pmsis.h>

void ext_pi_cl_team_barrier() {
  pi_cl_team_barrier();
}

int ext_pi_core_id() {
  return pi_core_id();
}

int omp_get_thread_num() {
  return pi_core_id();
}

void ext_pi_cl_team_fork(int nb_cores, void (*entry)(void *), void *arg) {
  pi_cl_team_fork(nb_cores, entry, arg);
}

int ext_pi_cl_nb_cores() {
  return pi_cl_team_nb_cores();
}
