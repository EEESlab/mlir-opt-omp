/* Demo kernel for the slides — a data-parallel vector add.
 *
 *   #pragma omp parallel for   ==>   omp.parallel { omp.wsloop { ... } }
 *
 * One construct that exercises the whole tool in a single pipeline:
 *   - the FORK           (omp.parallel  -> __kmpc_fork_call / GOMP_parallel / ext_pi_cl_team_fork)
 *   - the WORK-SHARING   (omp.wsloop    -> the DSL's pre / invoke / post phases + implicit barrier)
 *
 * It lowers on ALL THREE runtimes, and `#pragma omp parallel for` is exactly
 * what the ClangIR front end already emits (see quick-compile/test.c), so the
 * demo can start from C and show the *entire* journey, ClangIR included:
 *
 *     vecadd.c --clang/ClangIR--> CIR --cir-opt--> MLIR(omp) --mlir-opt-omp(3 passes)--> runtime calls
 *
 * This is THE file shown on the slides; run-demo.sh dumps every stage.
 */
void vecadd(int n, const int *a, const int *b, int *c) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
}
