#!/usr/bin/env python3
"""Recover the plotted values from the paper's figures.

The figures are matplotlib EPS, so they are vector: every bar is a rectangle
with real coordinates and every axis tick carries its own position and label.
That makes the numbers recoverable exactly -- this reads the figures, it does
not estimate them from a picture.

How it works. Text is emitted as `/glyphname glyphshow` inside a translated
gsave, so a label is the concatenation of its glyph names and its position is
the translate. Numeric labels stacked at one x, to the left of the axes, are the
y ticks: two of them fix the affine map from device units to data units. Bars
are filled axis-aligned rectangles standing on the baseline; the modal bottom
edge identifies the baseline and so separates the bars from the legend swatches.
Series are told apart by fill colour, and paired to kernels by x order.

Usage:
    python3 extract_from_eps.py                 # rewrite reference.csv
    python3 extract_from_eps.py --check         # verify it still matches
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reference.csv")

# figure file -> (column prefix, what it measures, the paper's figure number)
FIGURES = [
    ("results_gomp_LARGE_FINAL.eps", "fig4", "speedup", "4"),
    ("results_iomp_LARGE_FINAL.eps", "fig5", "speedup", "5"),
    ("results_pulp.eps", "fig6", "speedup", "6"),
    ("results_pulp_sizes.eps", "fig7", "size_pct", "7"),
    ("unroll_speedup.eps", "fig8", "unroll_pct", "8"),
]

GLYPH = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "period": ".", "comma": ",", "minus": "-", "hyphen": "-", "uni2212": "-",
    "percent": "%", "space": " ", "parenleft": "(", "parenright": ")",
    "plus": "+", "endash": "-", "emdash": "-", "underscore": "_",
}
NUM = r"[-+]?\d+(?:\.\d+)?"

RECT = re.compile(
    rf"({NUM})\s+({NUM})\s+m\s+({NUM})\s+({NUM})\s+l\s+"
    rf"({NUM})\s+({NUM})\s+l\s+({NUM})\s+({NUM})\s+l\s+cl"
    r"(?P<tail>(?:.|\n){0,200}?)fill", re.M)
COLOUR = re.compile(rf"({NUM})\s+({NUM})\s+({NUM})\s+setrgbcolor")
TEXT = re.compile(
    rf"({NUM})\s+({NUM})\s+translate\s*\n(?:\s*{NUM}\s+rotate\s*\n)?"
    r"(?P<body>(?:[^g]|g(?!restore))*?)grestore", re.M)


def texts(src):
    """(anchor_x, anchor_y, string, width).

    The width comes from the glyph advances the file already carries: each
    glyph is placed with `<dx> 0 m /name glyphshow`, so the last dx is how far
    along the run the last glyph starts. It is needed because the x tick labels
    are rotated and anchored at the END of the text, which makes a long name
    start further left than a short one at the next tick along -- sorting on
    the anchor alone puts `floyd-warshall` before `3mm`.
    """
    out = []
    for m in TEXT.finditer(src):
        body = m.group("body")
        glyphs = re.findall(r"/([A-Za-z0-9_.]+)\s+glyphshow", body)
        if not glyphs:
            continue
        offsets = [float(d) for d in re.findall(rf"({NUM})\s+{NUM}\s+m\s*\n\s*/",
                                                body)]
        width = max(offsets) if offsets else 0.0
        out.append((float(m.group(1)), float(m.group(2)),
                    "".join(GLYPH.get(g, g) for g in glyphs), width))
    return out


def rects(src):
    out = []
    for m in RECT.finditer(src):
        xs = [float(m.group(i)) for i in (1, 3, 5, 7)]
        ys = [float(m.group(i)) for i in (2, 4, 6, 8)]
        ux, uy = sorted(set(xs)), sorted(set(ys))
        if len(ux) != 2 or len(uy) != 2:
            continue
        c = COLOUR.search(m.group("tail"))
        col = tuple(round(float(c.group(i)), 3) for i in (1, 2, 3)) if c else None
        out.append((ux[0], uy[0], ux[1], uy[1], col))
    return out


def mode(values, tol=0.01):
    best, count = None, 0
    for v in values:
        n = sum(1 for w in values if abs(w - v) <= tol)
        if n > count:
            best, count = v, n
    return best


def read_figure(path):
    src = open(path, encoding="latin-1").read()
    txt, rs = texts(src), rects(src)

    # The bars first: where they start is what separates the y axis from the
    # plot, and the axis is what the values are read against.
    coloured = [r for r in rs if (r[2] - r[0]) < 300 and r[4]]
    if not coloured:
        raise SystemExit(f"{path}: no bars found")
    axes_left = min(r[0] for r in coloured)

    # y ticks: every numeric label left of the bars. Not grouped by exact x --
    # tick labels are right-aligned, so "10" begins further left than "0" and
    # an exact-x match splits one axis into two.
    ticks = sorted((y, float(t.rstrip("%")))
                   for x, y, t, _w in txt
                   if x < axes_left and re.fullmatch(r"-?\d+(?:\.\d+)?%?", t))
    if len(ticks) < 2:
        raise SystemExit(f"{path}: need two y ticks, found {len(ticks)}")
    (y0, v0), (y1, v1) = ticks[0], ticks[-1]
    scale = (v1 - v0) / (y1 - y0)
    to_data = lambda dev: v0 + (dev - y0) * scale

    # Every tick must land on the line the two ends define, or the axis is not
    # linear and reading a height off it would be wrong. The tolerance is tied
    # to the axis range rather than to the value: the coordinates in the file
    # carry about six significant digits, so the reconstruction is good to
    # roughly a ten-thousandth of the span and no better.
    span = abs(v1 - v0) or 1.0
    for dev, val in ticks:
        if abs(to_data(dev) - val) > 1e-4 * span:
            raise SystemExit(
                f"{path}: y ticks are not collinear -- {val} sits at "
                f"{to_data(dev):.6f}")

    # Bars stand on the baseline; the legend swatches do not.
    candidates = [r for r in coloured if r[0] >= axes_left]
    baseline = mode([r[1] for r in candidates])
    bars = [r for r in candidates if abs(r[1] - baseline) <= 0.01
            or abs(r[3] - baseline) <= 0.01]

    # Two series, told apart by colour, each in x order.
    by_colour = {}
    for b in bars:
        by_colour.setdefault(b[4], []).append(b)
    for v in by_colour.values():
        v.sort(key=lambda r: r[0])

    # Kernel names are the labels under the plot. Two things are NOT kernel
    # names and both have to go: the axis title and the legend entries.
    #
    # Below every bar excludes the legend. Using the bars rather than the
    # baseline matters on the size figure, where the baseline is the zero line
    # and sits mid-plot. The axis title goes by having a space in it, which no
    # PolyBench kernel name does -- position cannot separate them, because the
    # tick labels are rotated 45 degrees and the leftmost ones are anchored
    # further left than the plot area starts.
    floor = min(r[1] for r in bars)
    raw = [(x, y, t, w) for x, y, t, w in txt
           if not re.fullmatch(r"-?\d+(?:\.\d+)?%?", t)
           and " " not in t and y < floor]

    # The labels are rotated 45 degrees and anchored at the END of the text, so
    # a long name starts further left AND lower than a short one at the next
    # tick along: sorting on the anchor x alone puts `floyd-warshall` before
    # `3mm`.
    #
    # No need to measure the text, though. At 45 degrees the anchor sits the
    # same distance left as it does below, so tick_x = x - y + c for one
    # constant c shared by every label -- and ordering by x - y is therefore
    # the true left-to-right order, whatever each name's length.
    names = sorted((x - y, t) for x, y, t, _w in raw)
    return to_data, by_colour, [n[1] for n in names], baseline


def series_values(to_data, bars, baseline):
    """Signed height: a bar below the baseline is a negative value."""
    out = []
    for b in bars:
        top = b[3] if abs(b[1] - baseline) <= 0.01 else b[1]
        out.append(round(to_data(top), 4))
    return out


def build():
    rows = {}
    order = []
    meta = []
    for fname, prefix, kind, number in FIGURES:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"  skip {fname} (missing)")
            continue
        to_data, by_colour, names, baseline = read_figure(path)
        cols = sorted(by_colour, key=lambda c: -len(by_colour[c]))
        native = series_values(to_data, by_colour[cols[0]], baseline)
        ours = series_values(to_data, by_colour[cols[1]], baseline) if len(cols) > 1 else []
        n = min(len(names), len(native), len(ours) if ours else len(native))
        if n != len(names):
            raise SystemExit(
                f"{fname}: {len(names)} labels but {len(native)}/"
                f"{len(ours)} bars -- the pairing would be guesswork")
        meta.append((fname, number, kind, n))
        # Keyed by name, never by position: figure 8 covers a subset of the
        # suite, so an index into one figure means something else in another.
        for i in range(n):
            k = names[i]
            if k not in rows:
                rows[k] = {}
                order.append(k)
            rows[k][f"{prefix}_native"] = native[i]
            if ours:
                rows[k][f"{prefix}_our"] = ours[i]
        print(f"  {fname}: {n} kernels, {len(cols)} series")

    fields = ["kernel"]
    for _, prefix, kind, _ in FIGURES:
        fields += [f"{prefix}_native", f"{prefix}_our"]
    fields = [f for f in fields
              if f == "kernel" or any(f in r for r in rows.values())]

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        f.write("# Values recovered from the paper's figures.\n")
        f.write("#\n")
        f.write("# Not estimates: the figures are vector (matplotlib EPS), so\n")
        f.write("# every bar is a rectangle with real coordinates and every axis\n")
        f.write("# tick carries its own position. extract_from_eps.py reads them\n")
        f.write("# back through the axis transform. Regenerate with:\n")
        f.write("#     python3 extract_from_eps.py\n")
        f.write("# and check they still match the figures with --check.\n")
        f.write("#\n")
        for fname, number, kind, n in meta:
            f.write(f"#   fig{number}: {kind:<10} {n:>2} kernels   {fname}\n")
        f.write("#\n")
        f.write("# Kernel order is the order the figures use, which is the suite\n")
        f.write("# order of ALL_KERNELS in lib/kernels.sh -- not alphabetical.\n")
        f.write("#\n")
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";",
                           extrasaction="ignore")
        w.writeheader()
        for k in order:
            row = {"kernel": k}
            row.update(rows[k])
            w.writerow(row)
    print(f"  wrote {OUT} ({len(order)} kernels, {len(fields) - 1} columns)")
    return rows


def check():
    if not os.path.exists(OUT):
        raise SystemExit("reference.csv does not exist; run without --check")
    old = {}
    with open(OUT, encoding="utf-8") as f:
        lines = [l for l in f if not l.lstrip().startswith("#")]
    for r in csv.DictReader(lines, delimiter=";"):
        old[r["kernel"]] = r
    new = build()
    bad = 0
    for k, row in new.items():
        for col, v in row.items():
            o = old.get(k, {}).get(col)
            if o is None or abs(float(o) - v) > 1e-3:
                print(f"  DIFFERS {k}.{col}: {o} -> {v}")
                bad += 1
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        build()
