/* Fabric-controller boot code (GAP8 / PULP).
 *
 * On PULP, main() runs on the Fabric Controller (FC). The FC opens the cluster
 * and dispatches cluster_main() to it as a cluster task; the OpenMP team fork
 * (ext_pi_cl_team_fork, emitted by the pmsis lowering) then happens INSIDE the
 * cluster, on core 0. Same flow as quick-compile/pulp and the PolyBench-PULP
 * harness — kernel-agnostic, so it is reused verbatim for the demo.
 */
#include "pmsis.h"

#define FC_FREQ 250
#define CL_FREQ 175

struct pi_device cluster_dev;

void cluster_main();

static int test_entry() {
    struct pi_cluster_conf cl_conf;

    pi_freq_set(PI_FREQ_DOMAIN_FC, FC_FREQ * 1000000);

    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, &cl_conf);

    pi_freq_set(PI_FREQ_DOMAIN_CL, CL_FREQ * 1000000);

    if (pi_cluster_open(&cluster_dev)) {
        printf("Cluster open failed !\n");
        pmsis_exit(-1);
    }

    struct pi_cluster_task cl_task;
    pi_cluster_send_task_to_cl(&cluster_dev,
                               pi_cluster_task(&cl_task, cluster_main, NULL));

    pi_cluster_close(&cluster_dev);

    return 0;
}

static void test_kickoff(void *arg) {
    int ret = test_entry();
    pmsis_exit(ret);
}

int main() {
    return pmsis_kickoff((void *)test_kickoff);
}
