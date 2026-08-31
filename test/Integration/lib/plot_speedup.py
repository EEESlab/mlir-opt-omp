#!/usr/bin/env python3
# =============================================================================
# plot_speedup.py -- per-kernel bar charts from run_performance.sh output.
#
# Two metrics, same figure shape: one bar per toolchain per kernel, each
# measured against its OWN sequential baseline, so the two sides stay
# comparable even when their backends differ in code quality.
#
#   --metric speedup   (default) parallel speedup
#     native (ref)  =  ref_seq / ref_par     (stock OpenMP compiler)
#     our    (opt)  =  opt_seq / opt_par     (CIR/MLIR pipeline through this tool)
#
#     These are exactly the speedup_native / speedup_opt columns that
#     run_performance.sh already writes, so the plot is consistent with the CSV
#     and the console summary. If those columns are missing/NA the value is
#     recomputed from the raw cycle columns.
#
#   --metric size      binary size change, parallel against sequential
#     native (ref)  =  size_ref_par / size_ref_seq - 1
#     our    (opt)  =  size_opt_par / size_opt_seq - 1
#
#     What parallelising costs in bytes of linked ELF. Signed: a toolchain
#     whose parallel build is smaller than its sequential one plots below zero.
#     The size_* columns are written only for RUNTIME=pmsis (the footprint of
#     an embedded target is what makes them worth reporting), so this metric
#     needs a pmsis CSV.
#
# Kernels with no usable data for the chosen metric are dropped.
#
# The reference (native) bar is labelled by runtime, matching the paper figures:
#     iomp    -> "Clang frontend"
#     libgomp -> "GCC frontend"
#     pmsis   -> "PULP-SDK GCC"   (native OpenMP on GAP8/gvsoc)
#
# Usage:
#     python3 plot_speedup.py <results.csv> [output.png] [options]
#
# Options:
#     --metric {speedup,size}    what to plot (default: speedup)
#     --runtime {iomp,libgomp,pmsis}   picks the native bar's legend label
#     --title "..."              optional chart title (default: none, paper style)
#     --our-label "..."          override the "Our" legend label (default: "Our")
#
# Output format follows the extension of <output.png> (.png, .pdf, .svg, ...);
# .pdf gives a vector figure suitable for a paper. Defaults to <csv>.png, or
# <csv>-size.png for --metric size, so the two figures never overwrite each
# other when both are rendered from the same CSV.
#
# Requires: matplotlib, numpy  (pip install matplotlib numpy)
# =============================================================================

import argparse
import csv
import math
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe: no X display needed (servers, WSL, CI)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402

# Legend label for the reference/native bar, per runtime (matches the figures).
REF_LABEL = {
    "iomp": "Clang frontend",
    "libgomp": "GCC frontend",
    "pmsis": "PULP-SDK GCC",
}

# Bar styling: solid blue reference, hatched orange "Our".
REF_COLOR = "#4c78b4"
REF_EDGE = "#2f4f7a"
OUR_COLOR = "#c55a11"
OUR_EDGE = "#7a3708"
OUR_HATCH = "///"


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Plot parallel speedup or binary size change from "
        "run_performance.sh CSV output."
    )
    p.add_argument("csv", help="results_performance.csv produced by run_performance.sh")
    p.add_argument(
        "out",
        nargs="?",
        help="output image (default: <csv>.png, or <csv>-size.png for "
        "--metric size). Format follows the extension (.png/.pdf/.svg).",
    )
    p.add_argument(
        "--metric",
        choices=("speedup", "size"),
        default="speedup",
        help="speedup (default) or binary size change; size needs a pmsis CSV",
    )
    p.add_argument(
        "--runtime",
        choices=sorted(REF_LABEL),
        help="runtime used for the run; sets the native bar's legend label",
    )
    p.add_argument("--title", default=None, help="optional chart title")
    p.add_argument(
        "--our-label", default="Our", help='legend label for our bars (default: "Our")'
    )
    return p.parse_args(argv)


def short_name(kernel):
    """gemm-omp -> gemm : the display name used on the x axis."""
    return kernel[:-4] if kernel.endswith("-omp") else kernel


def _num(value):
    """Parse a CSV cell to a positive float, or None for NA/blank/non-positive."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() == "NA":
        return None
    try:
        f = float(value)
    except ValueError:
        return None
    return f if f > 0 else None


def speedup(row, ratio_col, seq_col, par_col):
    """Prefer the precomputed ratio column; else recompute seq/par from cycles."""
    r = _num(row.get(ratio_col))
    if r is not None:
        return r
    seq, par = _num(row.get(seq_col)), _num(row.get(par_col))
    if seq is not None and par is not None:
        return seq / par
    return None


def size_change(row, seq_col, par_col):
    """Percent change in linked-ELF bytes, the parallel build against its own
    sequential one.

    Signed on purpose. A parallel build that comes out *smaller* than its
    sequential counterpart is a real outcome, not a bad measurement, so it has
    to reach the plot as a negative bar rather than be dropped the way a
    missing cell is. Only the two inputs go through _num, which rejects the
    non-positive: a byte count of zero is a failed link, the result is free to
    be negative.
    """
    seq, par = _num(row.get(seq_col)), _num(row.get(par_col))
    if seq is None or par is None:
        return None
    return (par / seq - 1.0) * 100.0


def load_rows(csv_path, metric):
    kernels, native, our = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = (row.get("kernel") or "").strip()
            if not name or name == "GEOMEAN":  # skip the summary row
                continue
            if metric == "size":
                sn = size_change(row, "size_ref_seq", "size_ref_par")
                so = size_change(row, "size_opt_seq", "size_opt_par")
            else:
                sn = speedup(row, "speedup_native", "ref_seq_cyc", "ref_par_cyc")
                so = speedup(row, "speedup_opt", "opt_seq_cyc", "opt_par_cyc")
            if sn is None or so is None:  # skipped / failed kernel
                continue
            kernels.append(short_name(name))
            native.append(sn)
            our.append(so)
    return kernels, native, our


def make_plot(kernels, native, our, args):
    x = np.arange(len(kernels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(8, len(kernels) * 0.55 + 2), 5))

    ref_label = REF_LABEL.get(args.runtime, "Native")
    ax.bar(
        x - width / 2, native, width,
        label=ref_label, color=REF_COLOR, edgecolor=REF_EDGE, linewidth=0.6,
    )
    ax.bar(
        x + width / 2, our, width,
        label=args.our_label, color=OUR_COLOR, edgecolor=OUR_EDGE,
        linewidth=0.6, hatch=OUR_HATCH,
    )

    if args.title:
        ax.set_title(args.title)

    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=45, ha="right", fontsize=9)
    ax.set_xlim(-0.6, len(kernels) - 0.4)

    if args.metric == "size":
        # A signed quantity: the axis has to show zero, and show it as the
        # baseline rather than as the floor, or a shrinking binary reads as a
        # missing bar. The span is padded on whichever sides the data uses.
        ax.set_ylabel("Binary size change")
        lo = min(0.0, min(native), min(our))
        hi = max(0.0, max(native), max(our))
        pad = max((hi - lo) * 0.12, 0.05)
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
        ax.axhline(0, color="#444444", linewidth=0.8, zorder=2)
    else:
        ax.set_ylabel("Parallel speedup")
        ymax = max(max(native), max(our))
        ax.set_ylim(0, math.ceil(ymax * 1.08))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", framealpha=0.95)

    fig.tight_layout()
    return fig


def main(argv):
    args = parse_args(argv)
    # The two metrics come from one CSV, so their default names have to differ
    # or rendering both silently leaves only the second.
    suffix = "-size" if args.metric == "size" else ""
    out = args.out or (
        args.csv[:-4] + suffix + ".png"
        if args.csv.endswith(".csv")
        else args.csv + suffix + ".png"
    )

    kernels, native, our = load_rows(args.csv, args.metric)
    if not kernels:
        if args.metric == "size":
            sys.exit(
                f"plot_speedup: no size data in {args.csv}: the size_* columns "
                "are written only by RUNTIME=pmsis runs"
            )
        sys.exit(f"plot_speedup: no plottable kernels in {args.csv}")

    fig = make_plot(kernels, native, our, args)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}  ({len(kernels)} kernels)")


if __name__ == "__main__":
    main(sys.argv[1:])
