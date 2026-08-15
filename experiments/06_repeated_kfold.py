"""Experiment 6 - leaky-vs-clean under repeated stratified k-fold.

Replaces the single 80/20 split of Experiment 2 with repeated stratified
k-fold, reporting every number as a mean with a 95% interval. This closes the
methodological gap the paper criticises in others: no published study reports
k-fold on Edge-IIoTset.

Run:
    python experiments/06_repeated_kfold.py --subset ml --splits 5 --repeats 3
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from agriedge.config import results_dir
from agriedge.data.curated import load_curated
from agriedge.data.recipes import CleanConfig, prepare_clean, prepare_readme
from agriedge.evaluation.repeated import pivot_formatted, repeated_kfold, to_frame
from agriedge.models.zoo import MODELS


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("ml", "dnn"), default="ml")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        frame = load_curated(args.subset, as_strings=True, nrows=args.nrows)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total_folds = args.splits * args.repeats
    print(
        f"loaded {args.subset}: {len(frame):,} rows | "
        f"{args.splits}-fold x {args.repeats} repeats = {total_folds} fits per model\n"
    )

    scores = []

    print("--- Protocol A: recipe as distributed ---")
    leaky = prepare_readme(frame)
    print(f"  rows={leaky.n_rows:,} features={leaky.n_features}")
    try:
        scores.extend(
            repeated_kfold(
                leaky.features, leaky.binary, MODELS,
                protocol="readme", n_splits=args.splits, n_repeats=args.repeats,
            )
        )
    except ValueError as exc:
        print(f"error in protocol A: {exc}", file=sys.stderr)
        return 1

    print("\n--- Protocol B: corrected protocol ---")
    clean = prepare_clean(frame, CleanConfig())
    print(f"  rows={clean.n_rows:,} features={clean.n_features}")
    try:
        scores.extend(
            repeated_kfold(
                clean.features, clean.binary, MODELS,
                protocol="clean", n_splits=args.splits, n_repeats=args.repeats,
            )
        )
    except ValueError as exc:
        print(f"error in protocol B: {exc}", file=sys.stderr)
        return 1

    aggregate = tuple(scores)
    pd.set_option("display.width", 220, "display.max_columns", 20)

    for metric in ("accuracy", "f1_macro", "balanced_accuracy"):
        print(f"\n=== {metric} (mean ± 95% CI half-width, {total_folds} folds) ===")
        print(pivot_formatted(aggregate, metric).to_string())

    table = to_frame(aggregate)
    out = results_dir() / f"06_repeated_kfold_{args.subset}"
    table.to_csv(f"{out}.csv", index=False)
    print(f"\nresults written to {out}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
