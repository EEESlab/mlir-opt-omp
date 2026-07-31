# Machine setups

Working `local.env` files for the machines this project is developed on, kept
as references. The paths in them point at specific user accounts and will not
exist on your machine.

**Setting up for the first time? Use the generic template:**

```sh
cp local.env.example local.env
```

If one of these matches your setup, copy it from the repo root:

```sh
cp docs/setups/lucap-wsl.env local.env
```

| File | Machine | Runtimes |
|---|---|---|
| `lucap-wsl.env` | WSL dev box | `iomp`, `libgomp` |
| `lucap-workstation.env` | Lab workstation (Ubuntu) | `iomp`, `libgomp`, `pmsis` |
| `pulp-tagliavini.env` | PULP/gvsoc setup | `pmsis` |

These cover *where the tools are*. What the Integration tests run is separate
and optional — see
[`test/Integration/run.env.example`](../../test/Integration/run.env.example).
Every variable and its default is documented in
[`test/Integration/README.md`](../../test/Integration/README.md#configuration-reference).
