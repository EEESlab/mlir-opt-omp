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
#   4. ABSOLUTE    speedups against reference/expected-from-paper.csv. Reported
#                  for orientation and explicitly NOT a verdict: those values
#                  were read off the published charts by eye, and they were
#                  measured on other hardware, so a difference is two kinds of
#                  noise before it is ever a finding.
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

# Kernels the paper calls out by name, and which way it expects them to go.
# "ahead" means our speedup should beat the native one, "behind" the reverse.
# Named kernels are the checks that transfer best after the exact numbers,
# because they are claims about a mechanism rather than about a measurement.
KNOWN_OUTLIERS = {
    "libgomp": ("section 4.2", "ahead of", ["doitgen"]),
    "pmsis": ("section 4.3", "behind",
              ["floyd-warshall", "deriche", "nussinov"]),
}

# How far the parallel and sequential deficits may differ before the slowdown
# stops being uniform. Below this they are the same number, which is what makes
# it a codegen fact rather than a parallelisation one.
UNIFORM_TOLERANCE = 0.05

# Section 4.3: "the increase remains below 0.7% in all instances".
SIZE_BOUND_PCT = 0.7

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
        default=os.path.join(here, os.pardir, "reference",
                             "expected-from-paper.csv"),
        help="reference CSV (default: ../reference/expected-from-paper.csv)",
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
    print("   Absolute runtime against native: {:.3f}x parallel, "
          "{:.3f}x sequential.".format(gpar, gseq))
    if abs(gpar - gseq) <= UNIFORM_TOLERANCE:
        print("   The two agree, so the deficit is uniform: it is backend code")
        print("   quality rather than parallelisation, and it cancels in the")
        print("   ratio above. That is why the driver's summary can read {:.2f}"
              .format(gpar))
        print("   while parity reads about 1.00; they measure different things.")
    else:
        print("   The two DISAGREE, so the gap is not uniform: {:.3f}x of it is"
              .format(gpar / gseq if gseq else 0))
        print("   specific to the parallel cells and is not explained by")
        print("   backend code quality. That is worth looking into.")


def check_outliers(run, runtime, strict):
    """The kernels the paper names, and which way it expects each to go.

    These are claims about a mechanism — GCC leaving a loop-invariant guard
    inside doitgen's region that LLVM unswitches away, PULP GCC emitting
    cheaper barrier sequences — so unlike a speedup value they should hold on
    any machine. A named kernel that stops reproducing is not necessarily a
    regression: it can equally mean the sentence in the paper has gone out of
    date, which is worth knowing before the paper is submitted.
    """
    if runtime not in KNOWN_OUTLIERS:
        print("2. NAMED KERNELS - none for this runtime, skipped")
        print()
        return True

    section, direction, names = KNOWN_OUTLIERS[runtime]
    print("2. NAMED KERNELS - {} expects ours {} the native compiler here"
          .format(section, direction))
    print()

    reproduced, checked = 0, 0
    for name in names:
        row = run.get(name)
        if row is None:
            print("   {:<18}not in this run".format(name))
            continue
        native, our = num(row.get("speedup_native")), num(row.get("speedup_opt"))
        if native is None or our is None or native <= 0:
            print("   {:<18}no usable numbers".format(name))
            continue
        checked += 1
        ratio = our / native
        # A kernel within a few percent either way has simply stopped being an
        # outlier, which is a third outcome and not a failure.
        if abs(ratio - 1.0) <= PARITY_CLOSE:
            verdict = "at parity now"
        elif (ratio > 1.0) == direction.startswith("ahead"):
            verdict = "as documented"
            reproduced += 1
        else:
            verdict = "REVERSED, ours " + ("ahead" if ratio > 1 else "behind")
        print("   {:<18}native {:>6.2f}   ours {:>6.2f}   {:>6.3f}   {}".format(
            name, native, our, ratio, verdict))

    if checked:
        print()
        print("   {}/{} still reproduce. Where they no longer do, the paper's"
              .format(reproduced, checked))
        print("   text is the thing to revisit, not necessarily the compiler.")
    print()
    # Never fatal: a named kernel drifting is information about the prose.
    return True


def check_size(run, runtime, strict):
    """Section 4.3: below 0.7% in all instances. The size_* columns exist only
    on the PULP path, which is also the only target where the footprint is a
    result rather than a footnote."""
    if runtime != "pmsis":
        print("3. SIZE - pmsis only (the size_* columns), skipped")
        print()
        return True
    print("3. SIZE - section 4.3 expects every increase below {}%".format(
        SIZE_BOUND_PCT))
    over, checked = [], 0
    for name, row in run.items():
        seq, par = num(row.get("size_opt_seq")), num(row.get("size_opt_par"))
        if seq is None or par is None or seq <= 0:
            continue
        checked += 1
        pct = (par / seq - 1.0) * 100.0
        if pct > SIZE_BOUND_PCT:
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
    print()
    return not over or not strict


def check_absolute(run, reference, runtime):
    """Orientation only, and printed as one summary line rather than per kernel
    so it cannot be mistaken for a verdict."""
    _, our_col, figure = REFERENCE_COLUMNS[runtime]
    print("4. ABSOLUTE - against {}, for orientation only".format(figure))
    print("   chart readings, other hardware: differences here are expected")

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
    print("   {} kernels compared, geomean ours/paper {:.2f} "
          "({} of them above the published value)".format(
              len(ratios), geomean(ratios), above))
    print()


def explain_how(csv_path, runtime):
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
    if runtime == "pmsis":
        print("   size        size_opt_par / size_opt_seq - 1, against 0.7%.")
    print("   absolute    speedup_opt against {} of the reference file,".format(
        our_col))
    print("               which is {} read off by eye. Orientation only.".format(
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

    check_parity(run)
    ok_outliers = check_outliers(run, args.runtime, args.strict)
    ok_size = check_size(run, args.runtime, args.strict)
    if reference:
        check_absolute(run, reference, args.runtime)
    else:
        print("4. ABSOLUTE - skipped, no reference at {}".format(args.reference))
        print()
    explain_how(args.csv, args.runtime)

    if args.strict and not (ok_outliers and ok_size):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
