#!/usr/bin/env python3
"""
count_lines.py — Line counter for the mlir-opt-omp repository structure.
"""

import os
from pathlib import Path

# Target paths and directories defined in your repository structure
TARGET_PATHS = [
    "rules.dsl",
    "include/OmpLowering/IR",
    "include/OmpLowering/DSL",
    "include/OmpLowering/Transforms",
    "lib/DSL",
    "lib/IR",
    "lib/Transforms",
    "tools/mlir-opt-omp"
]

# File extensions to process, explicitly including .h and .hpp headers
VALID_EXTENSIONS = {
    ".h", ".hpp", ".cpp", ".td", ".dsl", ".mlir", ".c", 
    ".sh", ".env", ".md", ".py", ".txt", ".cmake"
}

def is_comment_or_blank(line: str, ext: str) -> tuple[bool, bool]:
    """Check if a line is blank or a single-line comment based on file type."""
    stripped = line.strip()
    if not stripped:
        return True, False  # is_blank, is_comment

    # Comment syntax rules per extension
    comment_prefixes = {
        ".h": ("//", "/*", "*"),
        ".hpp": ("//", "/*", "*"),
        ".cpp": ("//", "/*", "*"),
        ".c": ("//", "/*", "*"),
        ".td": ("//", "/*", "*"),
        ".dsl": ("//", "#"),
        ".mlir": ("//",),
        ".sh": ("#",),
        ".env": ("#",),
        ".py": ("#",),
        ".md": ("<!--",),
        ".cmake": ("#",),
    }

    prefixes = comment_prefixes.get(ext, ("//", "#"))
    if any(stripped.startswith(p) for p in prefixes):
        return False, True

    return False, False


def count_file_lines(file_path: Path) -> dict:
    """Count total, code, comment, and blank lines in a single file."""
    total = code = comment = blank = 0
    ext = file_path.suffix.lower()

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                is_b, is_c = is_comment_or_blank(line, ext)
                if is_b:
                    blank += 1
                elif is_c:
                    comment += 1
                else:
                    code += 1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return {"total": total, "code": code, "comment": comment, "blank": blank}


def main():
    root = Path(".")
    all_files = []

    # Collect files matching target paths and valid extensions
    for target in TARGET_PATHS:
        path = root / target
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            all_files.append(path)
        elif path.is_dir():
            for p in path.rglob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                    all_files.append(p)

    # De-duplicate and sort
    all_files = sorted(list(set(all_files)))

    if not all_files:
        print("No matching source/header files found. Run this from the project root.")
        return

    # Formatted terminal output
    header_fmt = "{:<55} | {:>8} | {:>8} | {:>8} | {:>8}"
    row_fmt    = "{:<55} | {:>8} | {:>8} | {:>8} | {:>8}"
    divider    = "-" * 99

    print(header_fmt.format("File", "Total", "Code", "Comment", "Blank"))
    print(divider)

    grand_totals = {"total": 0, "code": 0, "comment": 0, "blank": 0}

    for file_path in all_files:
        stats = count_file_lines(file_path)
        for k in grand_totals:
            grand_totals[k] += stats[k]

        rel_path = str(file_path)
        if len(rel_path) > 55:
            rel_path = "..." + rel_path[-52:]

        print(row_fmt.format(
            rel_path,
            stats["total"],
            stats["code"],
            stats["comment"],
            stats["blank"]
        ))

    print(divider)
    print(row_fmt.format(
        "TOTAL",
        grand_totals["total"],
        grand_totals["code"],
        grand_totals["comment"],
        grand_totals["blank"]
    ))

if __name__ == "__main__":
    main()