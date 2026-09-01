#!/usr/bin/env python3
# =============================================================================
# compare_to_reference.py -- put a finished run next to the paper's numbers.
#
#     python3 compare_to_reference.py <results_performance.csv> --runtime <r>
#
# It reports and does not judge. No row is called close, acceptable, at parity
# or reversed: the columns are printed side by side and what they mean is the
# reader's call. Two files supply the expected values, and both are named in
# the output so any figure here can be traced back:
#
#   reference/reference.csv   what the paper's figures plot, read out of the
#                             vector files, exact.
#   reference/claims.csv      what its text states, one row per sentence.
#
# --strict is the single exception, and it is opt-in: it exits 1 when a size
# claim with a `max` relation is exceeded, for a caller that wants a gate.
#
# Nothing is compared unless the run was made the way the figure was. A
# different dataset is a different program, and the barrier pass was not on
# when Figures 4-7 were measured, so those runs are reported as not comparable
# rather than tabulated against numbers they have nothing to do with. The
# driver passes --dataset, --threads and --barrier-elim; by hand they are
# optional, and what is not given is not checked.
# =============================================================================

import argparse
import csv
import math
import os
import sys

# runtime -> the pair of reference columns, and the figure they come from.
REFERENCE_COLUMNS = {
    "libgomp": ("fig4_native", "fig4_our", "fig4"),
    "iomp": ("fig5_native", "fig5_our", "fig5"),
    "pmsis": ("fig6_native", "fig6_our", "fig6"),
}

# The configuration each figure was measured in, from section 4.1: LARGE on the
# host with all 16 hardware threads, MINI on GAP8 whose core count the harness
# fixes. Dataset is a precondition -- a different one is a different problem,
# and the published columns would not be about the same program. Thread count
# is not: it changes the speedups but not what they are speedups of, so it is
# reported next to the figure's and left to the reader.
FIGURE_CONFIG = {
    "libgomp": ("LARGE_DATASET", 16),
    "iomp": ("LARGE_DATASET", 16),
    "pmsis": ("MINI_DATASET", None),
}

# Fallback for a checkout without reference/claims.csv: the rows that file
# holds, in the shape the loader hands on.
FALLBACK_CLAIMS = [
    ("doitgen-ahead", "4.2", "libgomp", "speedup_ratio", "doitgen",
     "above", 1.0, 0.10),
    ("floyd-warshall-behind", "4.3", "pmsis", "speedup_ratio",
     "floyd-warshall", "below", 1.0, 0.10),
    ("deriche-behind", "4.3", "pmsis", "speedup_ratio", "deriche",
     "below", 1.0, 0.10),
    ("nussinov-behind", "4.3", "pmsis", "speedup_ratio", "nussinov",
     "below", 1.0, 0.10),
    ("size-bound", "4.3", "pmsis", "size_increase_pct", "all",
     "max", 0.7, None),
]


def parse_args(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Put a run_performance.sh result next to the paper's.")
    p.add_argument("csv", help="results_performance.csv from a finished run")
    p.add_argument("--runtime", required=True, choices=sorted(REFERENCE_COLUMNS),
                   help="which runtime produced the CSV")
    p.add_argument("--reference",
                   default=os.path.join(here, os.pardir, "reference",
                                        "reference.csv"),
                   help="figure values (default: ../reference/reference.csv)")
    p.add_argument("--claims",
                   default=os.path.join(here, os.pardir, "reference",
                                        "claims.csv"),
                   help="paper claims (default: ../reference/claims.csv)")
    p.add_argument("--dataset",
                   help="DATASET the run used; without it the check is skipped")
    p.add_argument("--threads",
                   help="THREADS the run used, reported beside the figure's")
    p.add_argument("--barrier-elim", dest="barrier_elim",
                   help="BARRIER_ELIM the run used (0, 1 or both)")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if a 'max' claim is exceeded")
    return p.parse_args(argv)


def not_comparable(args):
    """Why this run cannot be put next to the figure, or None if it can."""
    dataset, _threads = FIGURE_CONFIG[args.runtime]
    figure = "Figure " + REFERENCE_COLUMNS[args.runtime][2][-1]
    if args.dataset and args.dataset != dataset:
        return ["this run used {}, {} was measured at {}.".format(
                    args.dataset, figure, dataset),
                "A different dataset is a different problem size, so the",
                "published columns would not be about the same program."]
    if args.barrier_elim and args.barrier_elim != "0":
        return ["BARRIER_ELIM={}, so --omp-barrier-elim was on.".format(
                    args.barrier_elim),
                "The figures were measured without it."]
    return None


def num(value):
    """A CSV cell as a float, or None for NA/blank/unparseable."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() == "NA":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def short_name(kernel):
    return kernel[:-4] if kernel.endswith("-omp") else kernel


def read_rows(path, skip_comments=False):
    """Rows as dicts, keyed by short kernel name, in file order."""
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    if skip_comments:
        lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
    rows = {}
    for row in csv.DictReader(lines, delimiter=";"):
        name = (row.get("kernel") or "").strip()
        if not name or name == "GEOMEAN":
            continue
        rows[short_name(name)] = row
    return rows


def read_claims(path):
    """The paper's prose claims. Falls back to FALLBACK_CLAIMS when absent."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    except OSError:
        return [
            {"claim": c, "section": sec, "runtime": rt, "metric": m,
             "subject": subj, "relation": rel, "value": val,
             "tolerance": tol, "checked_by": os.path.basename(__file__)}
            for c, sec, rt, m, subj, rel, val, tol in FALLBACK_CLAIMS
        ]
    claims = []
    for row in csv.DictReader(lines, delimiter=";"):
        name = (row.get("claim") or "").strip()
        if not name:
            continue
        row = {k: (v or "").strip() for k, v in row.items() if k}
        row["claim"] = name
        row["value"] = num(row.get("value"))
        row["tolerance"] = num(row.get("tolerance"))
        if row["value"] is not None:
            claims.append(row)
    return claims


def geomean(values):
    """The right average for ratios."""
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def cell(value, width=9, digits=3):
    if value is None:
        return "{:>{w}}".format("-", w=width)
    return "{:>{w}.{d}f}".format(value, w=width, d=digits)


def size_pct(row):
    seq, par = num(row.get("size_opt_seq")), num(row.get("size_opt_par"))
    if seq is None or par is None or seq <= 0:
        return None
    return (par / seq - 1.0) * 100.0


def has_size(run):
    return any(size_pct(r) is not None for r in run.values())


def table(run, reference, runtime):
    """One row per kernel: the run, the figure, and the ratios between them."""
    ref_native, ref_our, figure = REFERENCE_COLUMNS[runtime]
    show_size = has_size(run)

    head = "{:<16}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
        "kernel", "native", "ours", "ours/nat",
        figure + "_nat", figure + "_our", "ours/fig")
    if show_size:
        head += "{:>9}{:>9}".format("size%", "fig7%")
    print(head)
    print("-" * len(head))

    self_ratios, fig_ratios = [], []
    for name, row in run.items():
        ref = reference.get(name, {})
        native, ours = num(row.get("speedup_native")), num(row.get("speedup_opt"))
        f_native, f_ours = num(ref.get(ref_native)), num(ref.get(ref_our))
        self_r = ours / native if (ours and native and native > 0) else None
        fig_r = ours / f_ours if (ours and f_ours and f_ours > 0) else None
        if self_r:
            self_ratios.append(self_r)
        if fig_r:
            fig_ratios.append(fig_r)

        line = "{:<16}{}{}{}{}{}{}".format(
            name[:15], cell(native, digits=2), cell(ours, digits=2),
            cell(self_r), cell(f_native, digits=2), cell(f_ours, digits=2),
            cell(fig_r))
        if show_size:
            line += "{}{}".format(cell(size_pct(row), digits=3),
                                  cell(num(ref.get("fig7_size_our")), digits=3))
        print(line)

    print("-" * len(head))
    foot = "{:<16}{:>9}{:>9}{}{:>9}{:>9}{}".format(
        "geomean", "", "", cell(geomean(self_ratios)), "", "",
        cell(geomean(fig_ratios)))
    print(foot)
    print()

    par = geomean([num(r.get("opt_vs_native_par")) for r in run.values()])
    seq = geomean([num(r.get("opt_vs_native_seq")) for r in run.values()])
    if par or seq:
        print("opt_vs_native (absolute cycles, geomean)   "
              "parallel {}   sequential {}".format(
                  cell(par, width=1).strip(), cell(seq, width=1).strip()))
        print()


def claims_block(run, claims, runtime):
    """Every claim about this runtime, and the number this run puts beside it.

    'expects' is the paper's own wording as data; 'measured' is what came out.
    Whether the second satisfies the first is not decided here.
    """
    rows = [c for c in claims if c.get("runtime") in (runtime, "any")]
    if not rows:
        return True
    print("claims - what the paper states, and this run beside it")
    print("{:<9}{:<24}{:<22}{:<12}{}".format(
        "section", "claim", "expects", "measured", "source"))
    print("-" * 79)

    exceeded = False
    for c in sorted(rows, key=lambda c: (c["section"], c["claim"])):
        tol = "" if c["tolerance"] is None else " +/-{:g}".format(c["tolerance"])
        expects = "{} {:g}{}".format(c["relation"], c["value"], tol)
        measured, source = "-", c.get("checked_by") or "-"

        if c["metric"] == "speedup_ratio":
            row = run.get(c["subject"])
            native = num(row.get("speedup_native")) if row else None
            ours = num(row.get("speedup_opt")) if row else None
            if native and ours and native > 0:
                measured = "{:.3f}".format(ours / native)
            source = c["subject"] + " ours/nat"
        elif c["metric"] == "size_increase_pct":
            pcts = {n: size_pct(r) for n, r in run.items()}
            pcts = {n: p for n, p in pcts.items() if p is not None}
            if pcts:
                worst = max(pcts, key=pcts.get)
                measured = "{:.4f}".format(pcts[worst])
                source = "highest: {} ({} kernels)".format(worst, len(pcts))
                if c["relation"] == "max" and pcts[worst] > c["value"]:
                    exceeded = True

        print("{:<9}{:<24}{:<22}{:<12}{}".format(
            c["section"], c["claim"][:23], expects, measured, source))
    print()
    return not exceeded


def rel(path):
    """A path as short as it can be said from here."""
    try:
        short = os.path.relpath(path)
    except ValueError:            # a different drive, on Windows
        return os.path.normpath(path)
    full = os.path.normpath(path)
    return short if len(short) < len(full) else full


def legend(runtime, args, has_size):
    figure = REFERENCE_COLUMNS[runtime][2]
    print("columns")
    print("  native, ours   speedup of each variant against its OWN sequential"
          " cell")
    print("  {}_nat/_our  what Figure {} plots, from {}".format(
        figure, figure[-1], rel(args.reference)))
    if has_size:
        print("  size%          size_opt_par / size_opt_seq - 1")
        print("  fig7%          the same quantity in Figure 7")
    print("  claims         {}".format(rel(args.claims)))
    print("  the four cells behind a row: results/{}/<kernel>-omp/performance/"
          .format(runtime))
    print()


def config_line(args):
    """The run's configuration, and the figure's where the two can differ."""
    _dataset, threads = FIGURE_CONFIG[args.runtime]
    out = ""
    if args.dataset:
        out += "   dataset {}".format(args.dataset)
    if args.threads and threads:
        out += "   threads {}{}".format(
            args.threads,
            "" if str(args.threads) == str(threads)
            else " (figure: {})".format(threads))
    return out


def main(argv):
    args = parse_args(argv)
    try:
        run = read_rows(args.csv)
    except OSError as e:
        sys.exit("compare_to_reference: cannot read {}: {}".format(args.csv, e))
    if not run:
        sys.exit("compare_to_reference: no kernel rows in {}".format(args.csv))
    try:
        reference = read_rows(args.reference, skip_comments=True)
    except OSError:
        reference = {}
    claims = read_claims(args.claims)

    print()
    print("run      {}".format(args.csv))
    print("runtime  {}   kernels {}{}".format(
        args.runtime, len(run), config_line(args)))
    print()

    why = not_comparable(args)
    if why:
        print("not compared with the paper")
        for line in why:
            print("  " + line)
        print()
        return

    table(run, reference, args.runtime)
    ok = claims_block(run, claims, args.runtime)
    legend(args.runtime, args, has_size(run))

    if args.strict and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
