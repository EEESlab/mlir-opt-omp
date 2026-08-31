#!/usr/bin/env python3
# =============================================================================
# compare_to_reference.py -- read a finished run against what the paper claims.
#
# Usage:
#     python3 compare_to_reference.py <results_performance.csv> --runtime <r>
#
# The point of this script is to compare the things that survive a change of
# machine, and to say plainly which ones do not.
#
# A reviewer runs on a different CPU than the one the paper used, so absolute
# speedups are not comparable: gemm reaching 8x here and 5x there means nothing
# is wrong. What IS comparable is the relationship between the two bars --
# speedup_opt / speedup_native -- because that is a property of the compiler
# rather than of the hardware, and because it is what the paper actually claims:
# "our approach preserves performance across all benchmarks" (section 4.2).
#
# So the checks are ordered by how well they transfer:
#
#   1. PARITY      opt/native per kernel, from the run alone. Needs no
#                  reference file and no matching hardware, and it is the
#                  paper's central claim.
#   2. NAMED       the kernels the paper calls out: doitgen ahead on libgomp
#                  (section 4.2), floyd-warshall/deriche/nussinov behind on
#                  pmsis (section 4.3). Claims about a mechanism, so they
#                  transfer where a speedup value does not. Never fatal: one
#                  that stops reproducing usually means the prose has aged.
#   3. SIZE        section 4.3: the parallel build stays within 0.7% of the
#                  sequential one. An exact number from the prose, decided by
#                  the compiler, checkable on pmsis runs only.
#   4. ABSOLUTE    speedups against reference/reference.csv, which holds the
#                  values the paper's own figures plot -- recovered from the
#                  vector files, so exact. Still only orientation: they were
#                  measured on other hardware, and an absolute speedup does not
#                  survive a change of machine.
#
# Checks 2 and 3 read what they expect from reference/claims.csv rather than
# carrying it here, so that a sentence in the paper and the check that guards
# it stay one edit apart. The constants below are the fallback for a checkout
# without that file, and say what it used to hold.
#
# Exit status is 0 unless --strict is given, in which case only check 3 -- the
# one exact number -- can exit 1. The rest are readings, not assertions.
# =============================================================================

import argparse
import csv
import math
import os
import sys

# runtime -> the pair of columns in the reference CSV, and the figure it backs.
REFERENCE_COLUMNS = {
    "libgomp": ("fig4_native", "fig4_our", "Figure 4"),
    "iomp": ("fig5_native", "fig5_our", "Figure 5"),
    "pmsis": ("fig6_native", "fig6_our", "Figure 6"),
}

# Fallback for a checkout without reference/claims.csv: the same rows that
# file holds, in the shape the loader hands on. Kernels the paper calls out by
# name are the checks that transfer best after the exact numbers, because they
# are claims about a mechanism rather than about a measurement.
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

# How far the parallel and sequential deficits may differ before the slowdown
# stops being uniform. Below this they are the same number, which is what makes
# it a codegen fact rather than a parallelisation one.
UNIFORM_TOLERANCE = 0.05

# How far opt/native may stray before it is worth a look. Neither is a spec;
# they only decide which rows get printed as notable.
PARITY_CLOSE = 0.10   # within 10% -> parity
PARITY_WIDE = 0.25    # beyond 25% -> called out


def parse_args(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Compare a run_performance.sh result against the paper."
    )
    p.add_argument("csv", help="results_performance.csv from a finished run")
    p.add_argument(
        "--runtime", required=True, choices=sorted(REFERENCE_COLUMNS),
        help="which runtime produced the CSV",
    )
    p.add_argument(
        "--reference",
        default=os.path.join(here, os.pardir, "reference", "reference.csv"),
        help="reference CSV (default: ../reference/reference.csv)",
    )
    p.add_argument(
        "--claims",
        default=os.path.join(here, os.pardir, "reference", "claims.csv"),
        help="claims CSV (default: ../reference/claims.csv)",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="exit 1 if the size bound is exceeded (pmsis only)",
    )
    return p.parse_args(argv)


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
    """Rows as dicts, keyed by short kernel name.

    The reference file carries a '#' preamble saying it is not a measurement.
    That has to come off before csv sees it, or the first comment line becomes
    the header.
    """
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
    """The paper's prose claims, as a list of dicts.

    Falls back to FALLBACK_CLAIMS when the file is absent, so a checkout
    without it behaves exactly as this script did before the file existed.
    Rows the file carries but nothing here reads (the barrier counts, the
    unroll gains) are kept: section 6 reports them as coverage.
    """
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
        if row["value"] is None:
            continue
        claims.append(row)
    return claims


def select_claims(claims, metric, runtime=None):
    return [c for c in claims
            if c.get("metric") == metric
            and (runtime is None or c.get("runtime") in (runtime, "any"))]


def geomean(values):
    """The right average for ratios: the arithmetic mean of 0.5 and 2.0 claims
    a 25% gain where there is none."""
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def check_parity(run):
    print("1. PARITY - our speedup against the native one, same machine")
    print("   the paper's claim: performance is preserved across benchmarks")
    print()

    ratios, notable = [], []
    for name, row in run.items():
        native, our = num(row.get("speedup_native")), num(row.get("speedup_opt"))
        if native is None or our is None or native <= 0:
            continue
        ratio = our / native
        ratios.append(ratio)
        if abs(ratio - 1.0) > PARITY_CLOSE:
            notable.append((name, native, our, ratio))

    if not ratios:
        print("   no usable speedup columns in this CSV")
        print()
        return

    close = sum(1 for r in ratios if abs(r - 1.0) <= PARITY_CLOSE)
    print("   {}/{} kernels within 10% of parity, geomean {:.3f}".format(
        close, len(ratios), geomean(ratios)))

    if notable:
        print()
        print("   {:<18}{:>9}{:>9}{:>13}".format(
            "kernel", "native", "ours", "ours/native"))
        for name, native, our, ratio in sorted(notable, key=lambda t: t[3]):
            note = ""
            if abs(ratio - 1.0) > PARITY_WIDE:
                note = "  <-- " + ("faster" if ratio > 1 else "slower")
            print("   {:<18}{:>9.2f}{:>9.2f}{:>13.3f}{}".format(
                name, native, our, ratio, note))

    report_uniformity(run)
    print()


def report_uniformity(run):
    """Reconcile parity with the driver's own summary table.

    The summary prints opt/nat as absolute runtime, and on a backend that is
    simply slower than the native one that number sits well below 1.0 while
    parity sits at 1.0 — two figures a few lines apart that look like they
    contradict each other. They do not, and which of the two stories is true is
    decided by whether the deficit is the same in the sequential and parallel
    cells. If it is, it is codegen quality and cancels in a self-relative
    ratio; if it is not, the surplus is the parallelisation's own.
    """
    par = [v for v in (num(r.get("opt_vs_native_par")) for r in run.values()) if v]
    seq = [v for v in (num(r.get("opt_vs_native_seq")) for r in run.values()) if v]
    if not par or not seq:
        return
    gpar, gseq = geomean(par), geomean(seq)
    print()
    slower = (1 - gpar) * 100
    verb = "slower" if slower > 0 else "faster"
    print()
    print("   In absolute cycles our code is {:.0f}% {} than the native"
          .format(abs(slower), verb))
    print("   compiler's: {:.3f}x in the parallel runs, {:.3f}x in the"
          .format(gpar, gseq))
    print("   sequential ones.")
    print()
    if abs(gpar - gseq) <= UNIFORM_TOLERANCE:
        print("   Those two numbers are the same, which is the useful part: we")
        print("   are equally {} with one thread and with all of them. So it is"
              .format(verb))
        print("   the back end generating the code, not the way we parallelise")
        print("   it -- and because each speedup above is measured against that")
        print("   compiler's OWN sequential run, the {:.0f}% is in both the top"
              .format(abs(slower)))
        print("   and the bottom of the fraction and cancels out.")
        print()
        print("   That is why the driver's own summary can say {:.2f} while"
              .format(gpar))
        print("   parity here says about 1.00. Both are right; they answer")
        print("   different questions.")
    else:
        print("   Those two numbers DIFFER, and that is worth looking at: the")
        print("   gap is {:.3f}x wider in the parallel runs than in the"
              .format(gpar / gseq if gseq else 0))
        print("   sequential ones, so part of it belongs to the parallelisation")
        print("   itself and is not explained by code generation.")


def check_outliers(run, runtime, claims, strict):
    """The kernels the paper names, and which way it expects each to go.

    These are claims about a mechanism -- GCC leaving a loop-invariant guard
    inside doitgen's region that LLVM unswitches away, PULP GCC emitting
    cheaper barrier sequences -- so unlike a speedup value they should hold on
    any machine. A named kernel that stops reproducing is not necessarily a
    regression: it can equally mean the sentence in the paper has gone out of
    date, which is worth knowing before the paper is submitted.
    """
    rows = select_claims(claims, "speedup_ratio", runtime)
    if not rows:
        print("2. NAMED KERNELS - none claimed for this runtime, skipped")
        print()
        return True

    sections = sorted({r["section"] for r in rows})
    print("2. NAMED KERNELS - the kernels section{} {} call{} out by name"
          .format("" if len(sections) == 1 else "s",
                  ", ".join(sections),
                  "s" if len(sections) == 1 else ""))
    print()
    print("   {:<18}{:>9}{:>9}{:>9}{:>13}   {}".format(
        "kernel", "expects", "native", "ours", "ours/native", "verdict"))

    reproduced, checked = 0, 0
    for claim in rows:
        name = claim["subject"]
        wants = "ahead" if claim["relation"] == "above" else "behind"
        row = run.get(name)
        if row is None:
            print("   {:<18}{:>9}   not in this run".format(name, wants))
            continue
        native, our = num(row.get("speedup_native")), num(row.get("speedup_opt"))
        if native is None or our is None or native <= 0:
            print("   {:<18}{:>9}   no usable numbers".format(name, wants))
            continue
        checked += 1
        ratio = our / native
        target = claim["value"]
        band = claim["tolerance"] if claim["tolerance"] is not None else 0.0
        # A kernel within the claim's dead band either way has simply stopped
        # being an outlier, which is a third outcome and not a failure.
        if abs(ratio - target) <= band:
            verdict = "at parity now"
        elif (ratio > target) == (claim["relation"] == "above"):
            verdict = "as documented"
            reproduced += 1
        else:
            verdict = "REVERSED, ours " + ("ahead" if ratio > target else "behind")
        print("   {:<18}{:>9}{:>9.2f}{:>9.2f}{:>13.3f}   {}".format(
            name, wants, native, our, ratio, verdict))

    if checked:
        print()
        print("   {}/{} still reproduce. Where they no longer do, the paper's"
              .format(reproduced, checked))
        print("   text is the thing to revisit, not necessarily the compiler.")
    print()
    # Never fatal: a named kernel drifting is information about the prose.
    return True


def check_size(run, runtime, claims, reference, strict):
    """Section 4.3: below 0.7% in all instances. The size_* columns exist only
    on the PULP path, which is also the only target where the footprint is a
    result rather than a footnote."""
    rows = select_claims(claims, "size_increase_pct", runtime)
    if runtime != "pmsis" or not rows:
        print("3. SIZE - pmsis only (the size_* columns), skipped")
        print()
        return True
    bound = rows[0]["value"]
    print("3. SIZE - section {} expects every increase below {}%".format(
        rows[0]["section"], bound))
    over, checked, ours = [], 0, {}
    for name, row in run.items():
        seq, par = num(row.get("size_opt_seq")), num(row.get("size_opt_par"))
        if seq is None or par is None or seq <= 0:
            continue
        checked += 1
        pct = (par / seq - 1.0) * 100.0
        ours[name] = pct
        if pct > bound:
            over.append((name, pct))
    if not checked:
        print("   no size_* columns in this CSV - is this a pmsis run?")
        print()
        return True
    if over:
        print("   {}/{} kernels ABOVE the bound:".format(len(over), checked))
        for name, pct in sorted(over, key=lambda t: -t[1]):
            print("     {:<18}{:>8.2f}%".format(name, pct))
    else:
        print("   all {} kernels within the bound".format(checked))
    compare_size_to_figure(ours, reference)
    print()
    return not over or not strict


def compare_size_to_figure(ours, reference):
    """The one place where a published value should reproduce exactly.

    A speedup does not survive a change of machine, but a binary size is not a
    measurement of a machine at all: the same sources through the same
    toolchain give the same bytes. So unlike check 4, a divergence here is
    either a different SDK or a real change in what we emit -- and the second
    is worth knowing. Figure 7 is the only figure this applies to.
    """
    pairs = []
    for name, pct in ours.items():
        published = num((reference.get(name) or {}).get("fig7_size_our"))
        if published is not None:
            pairs.append((name, pct, published))
    if not pairs:
        return
    worst = max(pairs, key=lambda t: abs(t[1] - t[2]))
    close = sum(1 for _, pct, pub in pairs if abs(pct - pub) <= 0.05)
    print()
    print("   Against Figure 7 itself: {}/{} kernels within 0.05 points of"
          .format(close, len(pairs)))
    print("   the published value, furthest is {} ({:.3f}% here, {:.3f}%"
          .format(worst[0], worst[1], worst[2]))
    print("   published). This one should reproduce: a binary size is decided")
    print("   by the toolchain, not by the machine running it, so a gap means")
    print("   a different SDK or a real change in what we emit.")


def check_absolute(run, reference, runtime):
    """Orientation only, and printed as one summary line rather than per kernel
    so it cannot be mistaken for a verdict."""
    _, our_col, figure = REFERENCE_COLUMNS[runtime]
    print("4. ABSOLUTE - your speedups against the ones in {}".format(figure))
    print("   The published values are exact -- read out of the figure files --")
    print("   but they were measured on other hardware, so treat this as")
    print("   orientation rather than a verdict.")

    ratios = []
    for name, row in run.items():
        ref = reference.get(name)
        if ref is None:
            continue
        ours, ref_our = num(row.get("speedup_opt")), num(ref.get(our_col))
        if ours is None or ref_our is None or ref_our <= 0:
            continue
        ratios.append(ours / ref_our)

    if not ratios:
        print("   no kernels in common with the reference")
        print()
        return

    above = sum(1 for r in ratios if r > 1)
    g = geomean(ratios)
    print()
    print("   {} kernels compared. On average this run is {:.0f}% {} the".format(
        len(ratios), abs(g - 1) * 100, "above" if g >= 1 else "below"))
    print("   published figure ({} of the {} kernels came out higher).".format(
        above, len(ratios)))
    print("   Anything within roughly 20% either way is agreement at this")
    print("   resolution; what would be worth a second look is a kernel that")
    print("   moved by a factor, not a few percent.")
    print()


def explain_how(csv_path, runtime, claims):
    """Every number above, traced back to the columns it came from.

    A summary a reader cannot re-derive is a summary they have to trust, and
    the whole point of the artifact is that they should not have to.
    """
    native_col, our_col, figure = REFERENCE_COLUMNS[runtime]
    print("5. CHECKING THIS YOURSELF")
    print()
    print("   Everything above comes from one file, and every column in it is")
    print("   a plain number you can re-derive:")
    print()
    print("     {}".format(csv_path))
    print()
    print("   parity      speedup_opt / speedup_native, per row. Both columns")
    print("               are each variant against its OWN sequential cell, so")
    print("               the ratio is free of how fast the machine is.")
    print("   uniformity  the geomean of opt_vs_native_par against that of")
    print("               opt_vs_native_seq. Same number = codegen; different")
    print("               = something the parallel path is doing.")
    size_claims = select_claims(claims, "size_increase_pct", runtime)
    if size_claims:
        print("   size        size_opt_par / size_opt_seq - 1, against {}%,"
              .format(size_claims[0]["value"]))
        print("               and against fig7_size_our of reference.csv.")
    print("   absolute    speedup_opt against {} of reference.csv, which is".format(
        our_col))
    print("               what {} plots. Orientation only: other hardware.".format(
        figure))
    print()
    print("   Spot-check one kernel, no tooling:")
    print()
    print("     head -1 {0}".format(csv_path))
    print("     grep '^gemm' {0}".format(csv_path))
    print()
    print("   The four cells behind a row are kept next to it, so a surprising")
    print("   number can be run again by hand rather than taken on faith:")
    print("   results/{}/<kernel>-omp/performance/".format(runtime))
    print()
    print("   What checks 2 and 3 expect is not written into this script: it")
    print("   is in reference/claims.csv, one row per sentence of the paper.")
    print()


def report_uncovered(claims, runtime):
    """Claims about this runtime that nothing checks.

    Printing them is the point. A number in the paper with no driver behind it
    is exactly the kind that rots quietly, and a coverage gap is easier to act
    on when it is named than when it is simply absent.
    """
    mine = [c for c in claims
            if c.get("runtime") in (runtime, "any")
            and c.get("checked_by", "-") in ("", "-")]
    if not mine:
        return
    print("6. CLAIMS NOTHING CHECKS - {} for this runtime".format(len(mine)))
    print()
    for c in mine:
        print("   {:<22}section {:<5}{} {} {}".format(
            c["claim"], c["section"], c["metric"], c["relation"], c["value"]))
    print()


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
    print("Run:     {}".format(args.csv))
    print("Runtime: {}   kernels: {}".format(args.runtime, len(run)))
    print()

    claims = read_claims(args.claims)

    check_parity(run)
    ok_outliers = check_outliers(run, args.runtime, claims, args.strict)
    ok_size = check_size(run, args.runtime, claims, reference, args.strict)
    if reference:
        check_absolute(run, reference, args.runtime)
    else:
        print("4. ABSOLUTE - skipped, no reference at {}".format(args.reference))
        print()
    explain_how(args.csv, args.runtime, claims)
    report_uncovered(claims, args.runtime)

    if args.strict and not (ok_outliers and ok_size):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
