#!/usr/bin/env python3
# =============================================================================
# compare_to_reference.py -- comapre a run next to the paper's numbers.
#
#     python3 compare_to_reference.py <results_performance.csv> --runtime <r>
#
#   reference/reference.csv   what the paper's figures plot.
#
# A run made in a different configuration than the figure is reported as not
# comparable.
# =============================================================================

import argparse
import csv
import math
import os
import sys

# Colours, on the same rule as lib/common.sh: only when stdout is a terminal.
if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
    CYAN, RESET = "\033[0;36m", "\033[0m"
else:
    CYAN = RESET = ""

# runtime -> the pair of reference columns, and the figure they come from.
REFERENCE_COLUMNS = {
    "libgomp": ("fig4_native", "fig4_our", "fig4"),
    "iomp": ("fig5_native", "fig5_our", "fig5"),
    "pmsis": ("fig6_native", "fig6_our", "fig6"),
}

# The configuration each figure was measured in, from section 4.1: LARGE on the
# host with all 16 hardware threads, MINI on GAP8 whose core count the harness
# fixes. Dataset is a precondition; thread count is only reported next to the
# figure's.
FIGURE_CONFIG = {
    "libgomp": ("LARGE_DATASET", 16),
    "iomp": ("LARGE_DATASET", 16),
    "pmsis": ("MINI_DATASET", None),
}


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
    p.add_argument("--dataset",
                   help="DATASET the run used; without it the check is skipped")
    p.add_argument("--threads",
                   help="THREADS the run used, reported beside the figure's")
    p.add_argument("--barrier-elim", dest="barrier_elim",
                   help="BARRIER_ELIM the run used (0, 1 or both)")
    return p.parse_args(argv)


def not_comparable(args):
    """Why this run cannot be put next to the figure, or None if it can."""
    dataset, _threads = FIGURE_CONFIG[args.runtime]
    figure = "Figure " + REFERENCE_COLUMNS[args.runtime][2][-1]
    if args.dataset and args.dataset != dataset:
        return "this run used {}, {} was measured at {}.".format(
            args.dataset, figure, dataset)
    if args.barrier_elim and args.barrier_elim != "0":
        return "BARRIER_ELIM={}, the figures were measured without it.".format(
            args.barrier_elim)
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


def geomean(values):
    """Geometric mean, for ratios."""
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def cell(value, width=9, digits=3):
    if value is None:
        return "{:>{w}}".format("-", w=width)
    return "{:>{w}.{d}f}".format(value, w=width, d=digits)


def tint(text):
    """Colour a cell once it is padded, so the column widths still hold."""
    return CYAN + text + RESET if CYAN else text


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

    # opt-against-native, measured here and read off the figure: the same
    # quantity from the two sources, and the comparison the table exists to
    # make. They are the two coloured columns; the ruler is measured on the
    # plain text.
    head = "{:<16}{:>9}{:>9}{}{:>9}{:>9}{}".format(
        "kernel", "run_nat", "run_opt", tint("{:>9}".format("run_o/n")),
        figure + "_nat", figure + "_our",
        tint("{:>9}".format(figure + "_o/n")))
    width = 16 + 9 * 6
    if show_size:
        head += "{:>9}{:>9}".format("size%", "fig7%")
        width += 18
    print(head)
    print("-" * width)

    run_ratios, fig_ratios = [], []
    for name, row in run.items():
        ref = reference.get(name, {})
        native, opt = num(row.get("speedup_native")), num(row.get("speedup_opt"))
        f_native, f_our = num(ref.get(ref_native)), num(ref.get(ref_our))
        run_r = opt / native if (opt and native and native > 0) else None
        fig_r = (f_our / f_native
                 if (f_our and f_native and f_native > 0) else None)
        if run_r:
            run_ratios.append(run_r)
        if fig_r:
            fig_ratios.append(fig_r)

        line = "{:<16}{}{}{}{}{}{}".format(
            name[:15], cell(native, digits=2), cell(opt, digits=2),
            tint(cell(run_r)), cell(f_native, digits=2),
            cell(f_our, digits=2), tint(cell(fig_r)))
        if show_size:
            line += "{}{}".format(cell(size_pct(row), digits=3),
                                  cell(num(ref.get("fig7_size_our")), digits=3))
        print(line)

    print("-" * width)
    foot = "{:<16}{:>9}{:>9}{}{:>9}{:>9}{}".format(
        "geomean", "", "", tint(cell(geomean(run_ratios))), "", "",
        tint(cell(geomean(fig_ratios))))
    print(foot)
    print()


def rel(path):
    """The shorter of the relative and absolute path."""
    try:
        short = os.path.relpath(path)
    except ValueError:            # a different drive, on Windows
        return os.path.normpath(path)
    full = os.path.normpath(path)
    return short if len(short) < len(full) else full


def legend(runtime, args, has_size):
    figure = REFERENCE_COLUMNS[runtime][2]
    print("columns")
    print("  run_nat/_opt   measured by this run")
    print("  {}_nat/_our  what Figure {} plots, from {}".format(
        figure, figure[-1], rel(args.reference)))
    print("  run_o/n        run_opt / run_nat, this run's own ratio")
    print("  {}_o/n       {}_our / {}_nat, the same ratio in the figure"
          .format(figure, figure, figure))
    if has_size:
        print("  size%          size_opt_par / size_opt_seq - 1")
        print("  fig7%          the same value in Figure 7")
    print("  find run values: results/{}/<kernel>-omp/performance/"
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

    print()
    print("run      {}".format(args.csv))
    print("runtime  {}   kernels {}{}".format(
        args.runtime, len(run), config_line(args)))
    print()

    why = not_comparable(args)
    if why:
        print("not compared with the paper")
        print("  " + why)
        print()
        return

    table(run, reference, args.runtime)
    legend(args.runtime, args, has_size(run))


if __name__ == "__main__":
    main(sys.argv[1:])
