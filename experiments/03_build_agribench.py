"""Experiment 3 - build the AgriEdge benchmark and verify it is leakage-free.

Constructs the benchmark from the raw per-device captures under uniform
parsing, then re-runs the leakage audit against the result. A benchmark that
is fit for purpose must show no column capable of separating the classes on
its own.

Run:
    python experiments/03_build_agribench.py --out results/agriedge.parquet
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from agriedge.audit.leakage import audit_columns, duplicate_statistics, summarize
from agriedge.config import (
    BINARY_LABEL,
    LABEL_COLUMNS,
    MULTICLASS_LABEL,
    README_DUMMY_COLUMNS,
    results_dir,
)
from agriedge.data.agribench import (
    DEVICE_COLUMN,
    LAYER_COLUMN,
    BenchmarkConfig,
    build,
    feature_columns,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Parquet output path.")
    parser.add_argument("--rows-per-device", type=int, default=150_000)
    parser.add_argument("--rows-per-attack", type=int, default=60_000)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    config = BenchmarkConfig(
        max_rows_per_device=args.rows_per_device,
        max_rows_per_attack=args.rows_per_attack,
    )

    print("building AgriEdge benchmark from raw captures...")
    try:
        benchmark = build(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    features = feature_columns(benchmark)
    print(
        f"\nbenchmark: {len(benchmark):,} rows x {len(features)} features "
        f"(+{len(LABEL_COLUMNS)} labels, +2 provenance)"
    )

    print("\n=== Class balance ===")
    print(benchmark[MULTICLASS_LABEL].value_counts().to_string())
    binary = pd.to_numeric(benchmark[BINARY_LABEL])
    print(
        f"\nnormal={int((binary == 0).sum()):,}  attack={int((binary == 1).sum()):,}  "
        f"attack_rate={binary.mean():.4f}"
    )

    print("\n=== Device attribution (federated partitioning key) ===")
    print(benchmark[DEVICE_COLUMN].value_counts().to_string())
    print("\n=== Architectural layer ===")
    print(benchmark[LAYER_COLUMN].value_counts().to_string())

    print("\n=== Leakage re-audit of the constructed benchmark ===")
    auditable = tuple(c for c in README_DUMMY_COLUMNS if c in benchmark.columns)
    reports = audit_columns(benchmark, auditable, BINARY_LABEL)
    table = summarize(reports)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fully = table.loc[table["separation_rate"] >= 0.999, "column"].tolist()
    markers = table.loc[table["provenance_marker"], "column"].tolist()
    print(f"\nfully separating columns: {fully or 'none'}")
    print(f"provenance markers:       {markers or 'none'}")

    duplicates = duplicate_statistics(
        benchmark.drop(columns=list((DEVICE_COLUMN, LAYER_COLUMN))), LABEL_COLUMNS
    )
    print(
        f"exact duplicate rate:     {duplicates['exact_duplicate_rate']:.4f} "
        f"({int(duplicates['exact_duplicate_rows']):,} rows)"
    )

    out_path = args.out or (results_dir() / "agriedge_benchmark.parquet")
    benchmark.to_parquet(out_path, index=False)
    table.to_csv(results_dir() / "03_agribench_audit.csv", index=False)
    print(f"\nbenchmark written to {out_path}")

    if fully:
        print(
            "WARNING: benchmark still contains fully separating columns.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
