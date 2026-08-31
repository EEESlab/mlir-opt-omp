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
#   2. DOITGEN     section 4.2 singles it out as faster with our flow on
#                  libgomp. Qualitative, so it transfers.
#   3. SIZE        section 4.3: the parallel build stays within 0.7% of the
#                  sequential one. An exact number from the prose, decided by
#                  the compiler, checkable on pmsis runs only.
#   4. ABSOLUTE    speedups against reference/expected-from-paper.csv. Reported
#                  for orientation and explicitly NOT a verdict: those values
#                  were read off the published charts by eye, and they were
#                  measured on other hardware, so a difference is two kinds of
#                  noise before it is ever a finding.
#
# Exit status is 0 unless --strict is given, in which case a failed hard check
# (3, and 2 on libgomp) exits 1. Parity and the absolute comparison never fail
# the run: they are readings, not assertions.
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
        help="exit 1 if a hard check fails (size bound, doitgen on libgomp)",
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
    print()


def check_doitgen(run, runtime, strict):
    """Section 4.2 singles doitgen out as faster with our flow on libgomp: GCC
    leaves a loop-invariant guard inside the region that LLVM unswitches away.
    Qualitative, so unlike a speedup value it should hold on any machine."""
    if runtime != "libgomp":
        return True
    print("2. DOITGEN - section 4.2 expects ours ahead of GCC here")
    row = run.get("doitgen")
    if row is None:
        print("   not in this run, nothing to check")
        print()
        return True
    native, our = num(row.get("speedup_native")), num(row.get("speedup_opt"))
    if native is None or our is None or native <= 0:
        print("   no usable numbers for doitgen")
        print()
        return True
    ratio = our / native
    ok = ratio > 1.0
    print("   native {:.2f}   ours {:.2f}   ours/native {:.3f}   -> {}".format(
        native, our, ratio, "as expected" if ok else "NOT reproduced"))
    print()
    return ok or not strict


def check_size(run, runtime, strict):
    """Section 4.3: below 0.7% in all instances. The size_* columns exist only
    on the PULP path, which is also the only target where the footprint is a
    result rather than a footnote."""
    if runtime != "pmsis":
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
    ok_doitgen = check_doitgen(run, args.runtime, args.strict)
    ok_size = check_size(run, args.runtime, args.strict)
    if reference:
        check_absolute(run, reference, args.runtime)
    else:
        print("4. ABSOLUTE - skipped, no reference at {}".format(args.reference))
        print()

    if args.strict and not (ok_doitgen and ok_size):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
