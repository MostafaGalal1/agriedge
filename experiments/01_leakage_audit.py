"""Experiment 1 - audit the distributed preprocessing recipe for label leakage.

Reproduces the columns that ``Readme.txt`` instructs researchers to one-hot
encode, and measures how much of the label each one recovers on its own.

Run:
    python experiments/01_leakage_audit.py --subset ml
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from agriedge.audit.leakage import audit_columns, duplicate_statistics, summarize
from agriedge.config import (
    BINARY_LABEL,
    LABEL_COLUMNS,
    README_DUMMY_COLUMNS,
    results_dir,
)
from agriedge.data.curated import load_curated


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", choices=("ml", "dnn"), default="ml", help="Curated subset to audit."
    )
    parser.add_argument(
        "--nrows", type=int, default=None, help="Optional row cap for smoke tests."
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        frame = load_curated(args.subset, as_strings=True, nrows=args.nrows)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: could not load subset: {exc}", file=sys.stderr)
        return 1

    print(f"loaded {args.subset} subset: {frame.shape[0]:,} rows x {frame.shape[1]} cols")
    labels = pd.to_numeric(frame[BINARY_LABEL])
    print(f"  normal={int((labels == 0).sum()):,}  attack={int((labels == 1).sum()):,}\n")

    reports = audit_columns(frame, README_DUMMY_COLUMNS, BINARY_LABEL)
    table = summarize(reports)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== Per-column leakage (columns named in Readme.txt Step 5) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Zero-placeholder spelling by class ===")
    spelling_rows = []
    for report in reports:
        z = report.zero_spelling
        if z is None:
            continue
        spelling_rows.append(
            {
                "column": z.column,
                "'0' normal": z.string_zero_normal,
                "'0' attack": z.string_zero_attack,
                "'0.0' normal": z.float_zero_normal,
                "'0.0' attack": z.float_zero_attack,
                "provenance_marker": z.is_provenance_marker,
            }
        )
    spelling = pd.DataFrame(spelling_rows)
    print(spelling.to_string(index=False) if not spelling.empty else "(none)")

    duplicates = duplicate_statistics(frame, LABEL_COLUMNS)
    print("\n=== Duplicate rows (optimism under random splitting) ===")
    for key, value in duplicates.items():
        print(f"  {key}: {value:,.4f}" if "rate" in key else f"  {key}: {int(value):,}")

    out = results_dir() / f"01_leakage_audit_{args.subset}"
    table.to_csv(f"{out}_columns.csv", index=False)
    if not spelling.empty:
        spelling.to_csv(f"{out}_spelling.csv", index=False)
    with open(f"{out}_duplicates.json", "w", encoding="utf-8") as handle:
        json.dump(duplicates, handle, indent=2)

    fully = table.loc[table["separation_rate"] >= 0.999, "column"].tolist()
    print(f"\nfully separating columns ({len(fully)}): {fully}")
    print(f"results written to {out}_*.csv/.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
