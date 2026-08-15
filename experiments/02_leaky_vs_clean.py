"""Experiment 2 - how much of reported Edge-IIoTset performance is leakage?

Trains an identical model suite twice: once on features built by the recipe
distributed with the dataset, once on features built by the corrected protocol
proposed in this paper. The difference is the portion of published performance
attributable to a serialisation artifact rather than to intrusion detection.

Run:
    python experiments/02_leaky_vs_clean.py --subset ml
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from agriedge.config import RANDOM_SEED, results_dir
from agriedge.data.curated import load_curated
from agriedge.data.recipes import CleanConfig, prepare_clean, prepare_readme
from agriedge.evaluation.metrics import ScoreCard, fit_and_score, to_frame
from agriedge.models.zoo import MODELS


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("ml", "dnn"), default="ml")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--max-onehot-cardinality",
        type=int,
        default=CleanConfig().max_onehot_cardinality,
        help=(
            "Columns with more distinct tokens than this are encoded "
            "structurally instead of by identity. Lower values are stricter "
            "and prevent memorisation of literal payload strings."
        ),
    )
    return parser.parse_args(argv)


def evaluate_protocol(prepared, test_size: float) -> list[ScoreCard]:
    """Train every registered model on one prepared dataset."""
    x_train, x_test, y_train, y_test = train_test_split(
        prepared.features,
        prepared.binary,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=prepared.binary,
    )
    cards: list[ScoreCard] = []
    for spec in MODELS:
        try:
            card = fit_and_score(
                spec.build(),
                x_train,
                y_train,
                x_test,
                y_test,
                model_name=spec.name,
                protocol=prepared.protocol,
                split="random_stratified",
            )
        except (ValueError, MemoryError) as exc:
            print(f"  {spec.name}: skipped ({exc})", file=sys.stderr)
            continue
        cards.append(card)
        print(
            f"  {spec.name:22s} acc={card.accuracy:.4f} "
            f"f1_macro={card.f1_macro:.4f} bal_acc={card.balanced_accuracy:.4f}"
        )
    return cards


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        frame = load_curated(args.subset, as_strings=True, nrows=args.nrows)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"loaded {args.subset}: {frame.shape[0]:,} rows\n")

    print("--- Protocol A: recipe as distributed in Readme.txt ---")
    leaky = prepare_readme(frame)
    print(
        f"  rows={leaky.n_rows:,}  features={leaky.n_features}  "
        f"dummy_features={leaky.notes['n_dummy_features']}"
    )
    leaky_cards = evaluate_protocol(leaky, args.test_size)

    print("\n--- Protocol B: corrected protocol ---")
    clean = prepare_clean(
        frame, CleanConfig(max_onehot_cardinality=args.max_onehot_cardinality)
    )
    print(
        f"  rows={clean.n_rows:,}  features={clean.n_features}  "
        f"removed_as_leaking={list(clean.notes['leaking_columns_removed'])}"
    )
    print(
        f"  placeholder cells rewritten={clean.notes['placeholder_replacements']:,}  "
        f"duplicates_removed={clean.notes['duplicates_removed']:,}"
    )
    print(
        f"  structural encoding applied to={list(clean.notes['structural_columns'])}"
    )
    clean_cards = evaluate_protocol(clean, args.test_size)

    table = to_frame(leaky_cards + clean_cards)
    pd.set_option("display.width", 250, "display.max_columns", 30)

    pivot = table.pivot_table(
        index="model", columns="protocol", values=["accuracy", "f1_macro"]
    )
    pivot[("delta", "accuracy")] = (
        pivot[("accuracy", "readme")] - pivot[("accuracy", "clean")]
    )
    pivot[("delta", "f1_macro")] = (
        pivot[("f1_macro", "readme")] - pivot[("f1_macro", "clean")]
    )

    print("\n=== Leaky vs clean ===")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    out = results_dir() / f"02_leaky_vs_clean_{args.subset}"
    table.to_csv(f"{out}_scores.csv", index=False)
    pivot.to_csv(f"{out}_pivot.csv")
    print(f"\nresults written to {out}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
