#!/usr/bin/env python3
# =============================================================================
# plot_delta.py -- what --omp-barrier-elim saved, from the A/B run.
#
# Reads the CSV written by `BARRIER_ELIM=both ./run_performance.sh`: one row per
# kernel with the two timings, each with its standard deviation, and the saving
# already worked out as a percentage with the two deviations propagated.
#
# One horizontal bar per kernel, sorted, each with that error bar. The bar is
# the answer and the error bar is the caveat: a saving shorter than its own
# error bar is not a measurement, and on a host machine there are usually
# several of those. They are drawn in a muted colour rather than dropped —
# "too small to tell" is information too.
#
# Usage:
#     python3 plot_delta.py <results_performance_barrier-ab.csv> [out.png] [options]
#
# Options:
#     --runtime <name>   named in the axis label
#     --threads <n>      likewise
#
# Requires: matplotlib, numpy  (pip install matplotlib numpy)
# =============================================================================

import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe: no X display needed (servers, WSL, CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GAIN = "#009e73"
GAIN_EDGE = "#00654a"
LOSS = "#c55a11"
LOSS_EDGE = "#7a3708"
FAINT = "#b8b8b8"        # |saving| smaller than its own error bar
FAINT_EDGE = "#8a8a8a"
INK = "#3a3a3a"


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Plot the per-kernel saving from a BARRIER_ELIM=both run."
    )
    p.add_argument("csv", help="results_performance_barrier-ab.csv")
    p.add_argument("out", nargs="?", help="output image (default: <csv>.png)")
    p.add_argument("--runtime", default=None, help="runtime name, for the axis label")
    p.add_argument("--threads", default=None, help="thread count, for the axis label")
    return p.parse_args(argv)


def short_name(kernel):
    """gemm-omp -> gemm : the display name used on the axis."""
    return kernel[:-4] if kernel.endswith("-omp") else kernel


def _num(value):
    try:
        return float((value or "").strip())
    except ValueError:
        return None


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            name = (row.get("kernel") or "").strip()
            if not name or name == "TOTAL":     # the summary row
                continue
            delta, err = _num(row.get("delta_pct")), _num(row.get("delta_sd_pct"))
            if delta is None:                    # failed kernel, all NA
                continue
            rows.append((short_name(name), delta, err if err is not None else 0.0))
    return rows


def make_plot(rows, args):
    rows = sorted(rows, key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    y = np.arange(len(names))

    # Below its own error bar the sign means nothing, so the bar says so.
    def colour(v, e):
        if abs(v) <= e:
            return FAINT, FAINT_EDGE
        return (GAIN, GAIN_EDGE) if v > 0 else (LOSS, LOSS_EDGE)

    faces, edges = zip(*(colour(v, e) for v, e in zip(vals, errs)))

    fig, ax = plt.subplots(figsize=(8, max(2.4, len(names) * 0.26 + 1.2)))
    ax.barh(
        y, vals, 0.72, xerr=errs, color=list(faces), edgecolor=list(edges),
        linewidth=0.6, zorder=2,
        error_kw=dict(ecolor="#555555", elinewidth=0.8, capsize=2.2),
    )
    for yi, v, e in zip(y, vals, errs):
        off = e + 0.02 * (max(vals + [0]) - min(vals + [0]) or 1)
        ax.text(
            v + (off if v >= 0 else -off), yi, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=7.5, color=INK, zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_ylim(-0.8, len(names) - 0.2)

    label = "cicli risparmiati da --omp-barrier-elim (%)"
    if args.runtime:
        label += f"  —  {args.runtime}"
        if args.threads:
            label += f", {args.threads} thread"
    ax.set_xlabel(label)
    ax.axvline(0, color="#666666", linewidth=0.8, zorder=2)
    # Room for the value labels, which sit past the end of the error bars.
    lo = min([v - e for v, e in zip(vals, errs)] + [0.0])
    hi = max([v + e for v, e in zip(vals, errs)] + [0.0])
    pad = 0.16 * (hi - lo or 1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(
        0.995, 0.01,
        "in grigio: risparmio più piccolo del proprio errore",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="#6a6a6a",
    )

    fig.tight_layout()
    return fig


def main(argv):
    args = parse_args(argv)
    out = args.out or (
        args.csv[:-4] + ".png" if args.csv.endswith(".csv") else args.csv + ".png"
    )
    rows = load_rows(args.csv)
    if not rows:
        sys.exit(f"plot_delta: no plottable kernels in {args.csv}")
    fig = make_plot(rows, args)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}  ({len(rows)} kernel)")


if __name__ == "__main__":
    main(sys.argv[1:])
