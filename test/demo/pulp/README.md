# `test/demo/pulp` — the pmsis variant that runs on gvsoc

The host demo (`../run-demo.sh`) runs the kernel on `iomp` / `libgomp` and only
*lowers* it for `pmsis` (the toy driver is a host program). This folder closes
the loop: it takes the **same** kernel (`../vecadd.c`) all the way to a **GAP8**
executable and **runs it on the gvsoc simulator** — the "…and the same tool
targets embedded hardware" slide, with a real run behind it.

## How it works

```
../vecadd.c  #pragma omp parallel for
   │ clang -fclangir → cir-opt → mlir-opt-omp --omp-lower-runtime=pmsis
   │   → the pragma becomes  ext_pi_cl_team_fork(8, body, env) + a work-shared loop
   │ PULP_LLC (riscv32, +xpulpv)  →  kernel.o
   │
   ├─ pulp_main.c        FC boots, opens the cluster, dispatches cluster_main to core 0
   ├─ cluster_main.c     core 0 calls vecadd(...) (which team-forks) and prints
   ├─ interface-adapter.c  ext_pi_* shims → real PMSIS API (pi_cl_team_fork, ...)
   └─ Makefile           PULP-SDK app → link + run on gvsoc
```

`ext_pi_cl_team_fork` must be called from cluster core 0 — which is exactly
where the SDK dispatches `cluster_main`. This is a trimmed, self-contained copy
of `quick-compile/pulp`, so it needs **no external PolyBench-PULP harness**
(unlike `test/Integration`'s pmsis path).

## Run it (workstation only — needs the GAP SDK + gvsoc)

```bash
cp ../../Integration/config.env.lucap-workstation.example ../../Integration/config.env
./run-pulp.sh
```

`run-pulp.sh` reads the tool + PULP paths from `config.env`
(`LLVM_BIN`, `OMP_TOOL_BIN`, `RULES`, `INC_OMP`, `PULP_LLC`, `PULP_SDK_ENV`,
`PULP_TOOLCHAIN_BIN`), builds `kernel.o`, then links and runs on gvsoc.

Expected console output (same as every other runtime — `c[i] = 11*i`):

```
0
11
22
33
44
55
66
77
```

If the GAP SDK env is not sourced (`RULES_DIR` unset), the script stops after
producing `kernel.o` and tells you so.

## Files

| file | role |
|------|------|
| `run-pulp.sh`         | full pipeline: `../vecadd.c` → pmsis lowering → `kernel.o` → gvsoc run |
| `pulp_main.c`         | fabric-controller boot: open cluster, dispatch `cluster_main` |
| `cluster_main.c`      | cluster core-0 driver: call the kernel, print the result |
| `interface-adapter.c` | `ext_pi_*` shims → PMSIS API |
| `Makefile`            | PULP-SDK app that links `kernel.o` and runs on gvsoc |
