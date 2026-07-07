/* =============================================================================
 * omp_stubs.c — serial stubs for the OpenMP runtime API.
 *
 * Linked ONLY into the sequential builds (compiled without -fopenmp). Several
 * PolyBench/OMP kernels call omp_get_thread_num()/omp_get_num_threads()
 * unconditionally (not guarded by #ifdef _OPENMP), so without -fopenmp those
 * symbols are undefined and the link fails. These stubs make the sequential
 * baseline link and run as a true single-threaded program, with no OpenMP
 * runtime attached.
 *
 * Semantics follow the OpenMP spec's stub library: a single thread with id 0.
 * Signatures match <omp.h>; we deliberately do NOT include it, so the stub has
 * no header-path dependency. The parallel builds never link this file — they
 * get the real runtime via -fopenmp — so there is no symbol clash.
 * ===========================================================================*/

#include <sys/time.h>

int  omp_get_thread_num(void)   { return 0; }
int  omp_get_num_threads(void)  { return 1; }
int  omp_get_max_threads(void)  { return 1; }
int  omp_get_num_procs(void)    { return 1; }
int  omp_in_parallel(void)      { return 0; }
void omp_set_num_threads(int n) { (void)n; }
void omp_set_dynamic(int n)     { (void)n; }
int  omp_get_dynamic(void)      { return 0; }
void omp_set_nested(int n)      { (void)n; }
int  omp_get_nested(void)       { return 0; }

double omp_get_wtime(void)
{
    struct timeval t;
    gettimeofday(&t, 0);
    return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}
double omp_get_wtick(void) { return 1e-6; }
