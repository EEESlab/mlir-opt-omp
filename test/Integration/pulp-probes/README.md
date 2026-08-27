# PULP probes

Diagnostics for failures that only appear on the GAP8 cluster. Plain C against
the `ext_pi_*` shims the generated code calls, with nothing of mlir-opt-omp in
them — so a failure here belongs to the runtime, not the compiler.

## barrier-probe.c

Written for one open question: **seidel-2d hangs on PULP and the other 29
kernels do not.** It is also the only kernel in the suite where a sequential
loop sits between `#pragma omp parallel` and `#pragma omp for`:

```c
#pragma omp parallel private(t, w, i, j, ilo, ihi)
for (t = 0; t <= _PB_TSTEPS - 1; t++)
  for (w = 3; w <= 3 * (_PB_N - 2); w++)
    { ...; #pragma omp for
           for (i = ilo; i <= ihi; i++) ... }
```

so it is the only one that calls `ext_pi_cl_team_barrier` **repeatedly inside a
loop** — 176 times at MINI (`TSTEPS=2`, `N=32`). Everywhere else the team
barrier fires once per region entry. That is the difference the probe attacks.

### Running it

The drivers source the SDK environment and pass `platform=` themselves
(`lib/pulp.sh`); running `make` by hand means doing both:

```sh
source $PULP_SDK_ENV          # e.g. $GAP_SDK/configs/gap8_v3.sh — sets RULES_DIR
cd $PULP_APP_DIR
make clean all run platform=gvsoc \
  KERNEL_SRC=<repo>/test/Integration/pulp-probes/barrier-probe.c
```

Without the `source`, `RULES_DIR` is empty and the harness Makefile fails on
`include /pmsis_rules.mk`.

No `OMP_NATIVE` and no `OMP_OPT`: the probe *is* the kernel, and wants neither
gcc's OpenMP runtime nor a `kernel.o`.

### Reading the output

Every probe prints before it runs and again after. **The last line printed is
the probe that did not come back.**

| last line | what it means |
|---|---|
| `[1] MISMATCH` | `ext_pi_cl_nb_cores()` disagrees with the forked size. The chunk formula divides by it, so the work distribution is wrong. A `0` does not trap on RISC-V — it yields `-1` and every core computes an empty chunk. |
| hangs in `[2]` | The barrier shim cannot take consecutive calls — re-arming, or a counter not reset between uses. seidel-2d is then incidental: it is simply the only kernel that asks. **The likeliest outcome.** |
| hangs in `[3]` | The barrier assumes the cores arrive roughly in step. |
| hangs in `[4]` | A core whose chunk is empty does not reach the barrier. |
| hangs in `[5]`, `[2]`–`[4]` clean | The chunk arithmetic is wrong for seidel's bounds specifically. `reached t=/w=` says where. |
| `=== all probes returned ===` | The team barrier is not the problem. Look at the ELF and the stack next: the outlined body is larger than any other kernel's, and the cluster gives each core a small stack. |

### What is already excluded

Checked on the host before writing this, so the probe does not re-ask them:

- **The `private` clause.** The six privatizers become per-core `llvm.alloca` in
  `@outlined_parallel_0`; the packed env struct holds only `(i32, i32, ptr)`.
  `w` and `t` are not shared.
- **The barrier-elim pass.** It does not touch seidel-2d: `base=1, elim=1`. The
  `omp.wsloop` is nested, so `parentParallel` returns null and every rule
  declines.
- **Divergence at the barrier.** In the emitted IR the empty-chunk branch
  (`br i1 %38, ..., label %._crit_edge`) lands on the block that holds the
  barrier. Both paths converge on it.
- **Hoisted bounds.** `ext_pi_core_id` and `ext_pi_cl_nb_cores` stay inside the
  `w` loop body after `-O3`, so the chunk is recomputed per `w`.
- **The chunk arithmetic.** Transcribed and run over every `w` for `N=32`,
  `nc=8`: it partitions `[ilo, ihi]` exactly once, no gaps, no overlap. Trip
  counts run 1 to 15, so seven cores idle at the narrow end — `hi < lo` there,
  which is harmless under the `slt` guard the loop uses.
- **The lowering as a whole.** Lowered for libgomp and run on x86 at 1, 2, 4
  and 8 threads: terminates every time. That is the same
  `emit thread_bounds` path pmsis takes.
