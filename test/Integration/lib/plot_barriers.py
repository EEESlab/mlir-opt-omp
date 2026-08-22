#!/usr/bin/env python3
# =============================================================================
# plot_barriers.py -- team-barrier bar chart from the two barrier drivers.
#
# One bar per configuration per kernel, counting the barrier CALL SITES left in
# the emitted code. Reads either CSV and picks the bars from its header:
#
#   results_barrier_stats.csv       (run_barrier_stats.sh)
#       calls_base / calls_elim  -> 2 bars: ours without the pass, and with it.
#       Counted in the lowered MLIR, for $RUNTIME's barrier symbol.
#
#   results_barrier_vs_native.csv   (run_barrier_vs_native.sh)
#       clang / ours_baseline / ours_elim -> 3 bars, the stock compiler first.
#       Counted in LLVM IR after -O3, both sides at the same stage. iomp only.
#
# The two CSVs count the same thing at different stages, so a bar means the
# same in both figures and the colours are assigned per configuration, not per
# file: "ours" is orange in each, "ours + barrier elim" green in each.
#
# The explicit_removed / implicit_removed columns of the stats CSV are NOT
# plotted: they are the pass reporting on itself, while these bars are what
# survives to the emitted code. When the two disagree the CSV says MISMATCH,
# and the bars are still the trustworthy half.
#
# A count of zero draws no bar, so those are labelled: an empty slot then reads
# as "none left" rather than "no data". Kernels the driver could not build are
# dropped instead — they have no count at all.
#
# Usage:
#     python3 plot_barriers.py <results.csv> [output.png] [options]
#
# Options:
#     --runtime {iomp,libgomp,pmsis}   names the barrier symbol in the y label
#     --group-by <column>        block the x axis by that column's value
#     --only <column>=<value>    keep only the rows that match
#     --series <roles>           draw a subset of the bars (clang,base,elim)
#     --title "..."              optional chart title (default: none, paper style)
#
# --group-by pragma_form is what makes the whole-suite figure readable: in
# suite order the two regimes are interleaved, and the middle bar just looks
# worse than clang. Blocked, each half states one thing — on `split` clang and
# our baseline are identical and the pass takes one barrier per region off; on
# `combined` clang elides it in the front-end and the pass catches up. The
# column is optional, so asking for one the CSV does not have simply leaves
# the order alone.
#
# --only and --series cut that overview down to a figure that makes one point.
# The `combined` half is a tie, so a figure about what the pass adds says:
#
#     --only pragma_form=split                    clang, ours, ours+pass
#     --only pragma_form=split --series clang,elim    just the two that matter
#
# The legend totals follow the rows actually drawn, so the first of those
# reads "Clang (44)  Ours (44)  Ours + barrier elim (25)" — the baseline
# stating for itself that it is clang's, before the pass moves it.
#
# The gcc bar is the exception: it is read from the CSV but never drawn unless
# --series names it, because it is counted before -O3 and everything else
# after. Ask for it and you get its own figure, which is the only honest place
# for it:
#
#     --only pragma_form=split --series gcc,elim
#
# Output format follows the extension of <output.png> (.png, .pdf, .svg, ...);
# .pdf gives a vector figure suitable for a paper. Defaults to <csv>.png.
#
# Requires: matplotlib, numpy  (pip install matplotlib numpy)
# =============================================================================

import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe: no X display needed (servers, WSL, CI)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

# The call each runtime emits for a team barrier — same table as
# run_barrier_stats.sh, used to name the count in the y label.
BARRIER_SYM = {
    "iomp": "__kmpc_barrier",
    "libgomp": "GOMP_barrier",
    "pmsis": "ext_pi_cl_team_barrier",
}

# One entry per configuration a bar can show, keyed by the role rather than by
# the CSV column, so the same configuration keeps its colour across both
# figures. Blue/orange are the reference/ours pair plot_speedup.py already
# uses; the hatches carry the same distinction without colour, for print and
# for colour-blind readers.
SERIES = {
    "clang": dict(label="Clang", color="#4c78b4", edge="#2f4f7a", hatch=""),
    "gcc": dict(label="GCC", color="#7b5aa6", edge="#4a3466", hatch="\\\\\\"),
    "base": dict(label="Ours", color="#c55a11", edge="#7a3708", hatch="///"),
    "elim": dict(
        label="Ours + barrier elim", color="#009e73", edge="#00654a", hatch="xxx"
    ),
}

# Which column holds each role, per CSV. The first table whose columns are all
# present wins, so the file is recognised by its header alone.
LAYOUTS = [
    {"clang": "clang", "base": "ours_baseline", "elim": "ours_elim"},
    {"base": "calls_base", "elim": "calls_elim"},
]

# Bars that are read when the column is there but never drawn unless --series
# asks for them by name. gcc is counted before -O3 while everything else is
# counted after it (see run_barrier_vs_native.sh), so it belongs in a figure of
# its own rather than silently alongside bars from another stage.
OPTIONAL = {"gcc": "gcc_o0"}

ZERO_LABEL_COLOR = "#5a5a5a"
GROUP_LABEL_COLOR = "#3a3a3a"
GROUP_RULE_COLOR = "#9a9a9a"
# Height of a block label, in axes fractions: below the rotated kernel names.
GROUP_LABEL_Y = -0.30

# Left-to-right order for the blocks of --group-by, for values we know about.
# `split` leads: it is the half where the stock compiler keeps the barrier and
# the figure has something to show. Anything unlisted follows, in CSV order.
GROUP_ORDER = ["split", "combined"]

# What a block is called under the axis. `split`/`combined` name a column
# value, which says nothing to a reader who has not read the driver: spell out
# the directive each stands for instead. Any other value is used as it is.
GROUP_LABELS = {
    "split": "separate directives:  #pragma omp parallel { #pragma omp for }",
    "combined": "combined directive:  #pragma omp parallel for",
}


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Plot team-barrier counts from the barrier drivers' CSV output."
    )
    p.add_argument("csv", help="results_barrier_stats.csv or results_barrier_vs_native.csv")
    p.add_argument(
        "out",
        nargs="?",
        help="output image (default: <csv> with a .png extension). "
        "Format follows the extension (.png/.pdf/.svg).",
    )
    p.add_argument(
        "--runtime",
        choices=sorted(BARRIER_SYM),
        help="runtime the counts were taken for; names the symbol in the y label",
    )
    p.add_argument(
        "--group-by",
        default=None,
        metavar="COLUMN",
        help="block the x axis by this column's value, one labelled block per "
        "value, separated by a rule (e.g. pragma_form). Ignored when the CSV "
        "has no such column",
    )
    p.add_argument(
        "--only",
        default=None,
        metavar="COLUMN=VALUE",
        help="keep only the rows whose COLUMN holds VALUE, e.g. "
        "pragma_form=split. Unlike --group-by this one is an error when the "
        "column is missing: a filter that silently did nothing would draw "
        "every kernel under a caption saying otherwise",
    )
    p.add_argument(
        "--series",
        default=None,
        metavar="ROLES",
        help="comma-separated subset of the bars to draw (%s); "
        "default: all the CSV has" % ", ".join(SERIES),
    )
    p.add_argument("--title", default=None, help="optional chart title")
    return p.parse_args(argv)


def parse_only(spec):
    """--only COLUMN=VALUE -> (column, value); exits on a malformed one."""
    if spec is None:
        return None, None
    column, sep, value = spec.partition("=")
    if not sep or not column.strip():
        sys.exit(f"plot_barriers: --only wants COLUMN=VALUE, got '{spec}'")
    return column.strip(), value.strip()


def short_name(kernel):
    """gemm-omp -> gemm : the display name used on the x axis."""
    return kernel[:-4] if kernel.endswith("-omp") else kernel


def _count(value):
    """Parse a CSV cell to a barrier count. None for blank/NA — zero is a count."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() == "NA":
        return None
    try:
        n = int(value)
    except ValueError:
        return None
    return n if n >= 0 else None


def pick_layout(fieldnames):
    """The role -> column map matching this CSV's header, or None."""
    have = set(fieldnames or ())
    for layout in LAYOUTS:
        if set(layout.values()) <= have:
            return layout
    return None


def load_rows(csv_path, group_col=None, only_col=None, only_value=None):
    """Kernel names, one count list per role, and the group of each kernel.

    Rows the driver could not build have no counts and are dropped, as are the
    ones --only rules out. The group is None throughout unless --group-by named
    a column the CSV actually has.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        layout = pick_layout(reader.fieldnames)
        if layout is None:
            sys.exit(
                f"plot_barriers: {csv_path} has none of the expected column sets "
                f"({' | '.join(', '.join(l.values()) for l in LAYOUTS)})"
            )
        layout = dict(layout)
        layout.update(
            {r: c for r, c in OPTIONAL.items() if c in (reader.fieldnames or ())}
        )
        if group_col and group_col not in (reader.fieldnames or ()):
            group_col = None
        if only_col and only_col not in (reader.fieldnames or ()):
            sys.exit(f"plot_barriers: {csv_path} has no column '{only_col}'")

        kernels, groups = [], []
        counts = {role: [] for role in layout}
        for row in reader:
            name = (row.get("kernel") or "").strip()
            if not name:
                continue
            if only_col and (row.get(only_col) or "").strip() != only_value:
                continue
            values = {role: _count(row.get(col)) for role, col in layout.items()}
            if any(v is None for v in values.values()):  # ERROR row, no counts
                continue
            kernels.append(short_name(name))
            groups.append((row.get(group_col) or "").strip() if group_col else None)
            for role, v in values.items():
                counts[role].append(v)
    # SERIES fixes the drawing order; the CSV only says which roles are there.
    series = [(role, counts[role]) for role in SERIES if role in counts]
    return kernels, series, (groups if group_col else None)


def regroup(kernels, series, groups):
    """Reorder into one block per group value.

    Returns the reordered lists plus the blocks as (label, first, last) index
    spans. Order within a block is the CSV's, so a kernel keeps its neighbours.
    """
    seen = []
    for g in groups:
        if g not in seen:
            seen.append(g)
    order = [g for g in GROUP_ORDER if g in seen]
    order += [g for g in seen if g not in order]

    index = [i for g in order for i, gi in enumerate(groups) if gi == g]
    kernels = [kernels[i] for i in index]
    series = [(role, [values[i] for i in index]) for role, values in series]

    blocks, start = [], 0
    for g in order:
        n = sum(1 for gi in groups if gi == g)
        blocks.append((g, start, start + n - 1))
        start += n
    return kernels, series, blocks


def make_plot(kernels, series, blocks, args):
    x = np.arange(len(kernels))
    slot = 0.8 / len(series)
    width = slot * 0.92  # leave a gap between neighbouring bars

    per_kernel = 0.55 if len(series) < 3 else 0.62
    fig, ax = plt.subplots(figsize=(max(8, len(kernels) * per_kernel + 2), 5))

    for i, (role, values) in enumerate(series):
        style = SERIES[role]
        offset = (i - (len(series) - 1) / 2) * slot
        ax.bar(
            x + offset, values, width,
            label=f"{style['label']} ({sum(values)})",
            color=style["color"], edgecolor=style["edge"],
            linewidth=0.6, hatch=style["hatch"],
        )
        # A zero has no bar to see, so say so where the bar would have been.
        for xi, v in zip(x, values):
            if v == 0:
                ax.text(
                    xi + offset, 0.05, "0", ha="center", va="bottom",
                    fontsize=7, color=ZERO_LABEL_COLOR,
                )

    ylabel = "Team barrier call sites"
    # Naming the symbol is only honest while every bar counts the same one: the
    # gcc bar counts GOMP_barrier, the rest count whatever $RUNTIME emits.
    if args.runtime and not any(role == "gcc" for role, _ in series):
        ylabel += f" ({BARRIER_SYM[args.runtime]})"
    ax.set_ylabel(ylabel)
    if args.title:
        ax.set_title(args.title, pad=28)

    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=45, ha="right", fontsize=9)
    ax.set_xlim(-0.6, len(kernels) - 0.4)

    # One rule between blocks, and a label under each. The label sits below the
    # kernel names, where a second axis level belongs — putting it inside the
    # plot would mean finding empty sky above bars of every height.
    for label, first, last in blocks:
        if first:
            ax.axvline(
                first - 0.5, color=GROUP_RULE_COLOR, linewidth=0.9,
                linestyle=(0, (4, 3)),
            )
        n = last - first + 1
        ax.text(
            (first + last) / 2, GROUP_LABEL_Y,
            f"{GROUP_LABELS.get(label, label)}   —   "
            f"{n} kernel{'s' if n > 1 else ''}",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=10, color=GROUP_LABEL_COLOR,
        )

    ax.set_ylim(0, max(max(v) for _, v in series) + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Above the axes: the bars reach different heights per kernel, so no corner
    # is reliably free. The totals ride in the labels — they are the headline.
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01),
        ncol=len(series), frameon=False,
    )

    fig.tight_layout()
    return fig


def main(argv):
    args = parse_args(argv)
    out = args.out or (
        args.csv[:-4] + ".png" if args.csv.endswith(".csv") else args.csv + ".png"
    )

    only_col, only_value = parse_only(args.only)
    kernels, series, groups = load_rows(args.csv, args.group_by, only_col, only_value)
    if not kernels:
        sys.exit(f"plot_barriers: no plottable kernels in {args.csv}")

    if args.series:
        wanted = [r.strip() for r in args.series.split(",") if r.strip()]
        unknown = [r for r in wanted if r not in dict(series)]
        if unknown:
            sys.exit(
                f"plot_barriers: no {', '.join(unknown)} bar in {args.csv} "
                f"(it has {', '.join(role for role, _ in series)})"
            )
        series = [(role, values) for role, values in series if role in wanted]
    else:
        series = [(role, values) for role, values in series if role not in OPTIONAL]

    blocks = []
    if groups:
        kernels, series, blocks = regroup(kernels, series, groups)

    fig = make_plot(kernels, series, blocks, args)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}  ({len(kernels)} kernels)")


if __name__ == "__main__":
    main(sys.argv[1:])
