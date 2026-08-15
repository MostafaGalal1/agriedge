"""Experiment 4 - centralized detection and edge cost on the AgriEdge benchmark.

Establishes honest reference numbers on the corrected benchmark under two
splitting protocols:

* ``random_stratified`` - the split used throughout the literature.
* ``leave_one_device_out`` - train on four sensor types, test on the fifth.
  This is the deployment case that matters: a farm adds a sensor the detector
  was never trained on, and the detector must still work.

Each fitted model is then measured for on-gateway deployment cost.

Run:
    python experiments/04_agriedge_centralized.py --sample 400000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agriedge.config import RANDOM_SEED, results_dir
from agriedge.data.agribench import to_model_matrix
from agriedge.evaluation import edge
from agriedge.evaluation.metrics import ScoreCard, fit_and_score, to_frame
from agriedge.models.zoo import MODELS, edge_models


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default=str(results_dir() / "agriedge_benchmark.parquet"),
        help="Path to the parquet built by experiment 3.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row subsample, for a faster local run.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--holdout-device",
        default="Modbus",
        help="Device withheld for the leave-one-device-out split.",
    )
    return parser.parse_args(argv)


def random_split_scores(matrix, test_size: float) -> list[ScoreCard]:
    x_train, x_test, y_train, y_test = train_test_split(
        matrix.features,
        matrix.binary,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=matrix.binary,
    )
    cards = []
    for spec in MODELS:
        try:
            card = fit_and_score(
                spec.build(), x_train, y_train, x_test, y_test,
                model_name=spec.name, protocol="agriedge",
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


def leave_one_device_out_scores(matrix, holdout: str) -> list[ScoreCard]:
    """Withhold one sensor's normal traffic; attacks stay in both halves."""
    is_holdout = (matrix.devices == holdout).to_numpy()
    if not is_holdout.any():
        raise ValueError(
            f"Device {holdout!r} not present. Available: "
            f"{sorted(matrix.devices.unique())}"
        )

    is_attack = matrix.devices.str.startswith("attack:").to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    attack_to_test = rng.random(len(matrix.devices)) < 0.2

    test_mask = is_holdout | (is_attack & attack_to_test)
    train_mask = ~is_holdout & ~(is_attack & attack_to_test)

    x_train = matrix.features.loc[train_mask]
    y_train = matrix.binary.loc[train_mask]
    x_test = matrix.features.loc[test_mask]
    y_test = matrix.binary.loc[test_mask]

    print(
        f"  train={len(x_train):,} (no {holdout})  test={len(x_test):,} "
        f"(all {holdout} + 20% attacks)"
    )
    cards = []
    for spec in MODELS:
        try:
            card = fit_and_score(
                spec.build(), x_train, y_train, x_test, y_test,
                model_name=spec.name, protocol="agriedge",
                split=f"leave_out_{holdout}",
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


def edge_costs(matrix, test_size: float) -> pd.DataFrame:
    x_train, x_test, y_train, _ = train_test_split(
        matrix.features, matrix.binary, test_size=test_size,
        random_state=RANDOM_SEED, stratify=matrix.binary,
    )
    costs = []
    for spec in edge_models():
        model = spec.build()
        model.fit(x_train, y_train)
        costs.append(
            edge.measure(model, x_test.to_numpy(), model_name=spec.name)
        )
    return edge.to_frame(costs)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        frame = pd.read_parquet(args.benchmark)
    except (FileNotFoundError, OSError) as exc:
        print(f"error: could not read benchmark: {exc}", file=sys.stderr)
        return 1

    if args.sample and args.sample < len(frame):
        frame = frame.sample(n=args.sample, random_state=RANDOM_SEED)
        print(f"subsampled to {len(frame):,} rows")

    matrix = to_model_matrix(frame)
    print(f"model matrix: {matrix.shape[0]:,} rows x {matrix.shape[1]} features\n")

    print("--- Split A: random stratified (literature convention) ---")
    random_cards = random_split_scores(matrix, args.test_size)

    print(f"\n--- Split B: leave-one-device-out ({args.holdout_device}) ---")
    try:
        lodo_cards = leave_one_device_out_scores(matrix, args.holdout_device)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        lodo_cards = []

    table = to_frame(random_cards + lodo_cards)
    pd.set_option("display.width", 250, "display.max_columns", 30)

    if lodo_cards:
        pivot = table.pivot_table(
            index="model", columns="split", values=["f1_macro", "balanced_accuracy"]
        )
        print("\n=== Random split vs unseen device ===")
        print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n--- Edge deployment cost ---")
    costs = edge_costs(matrix, args.test_size)
    print(costs.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = results_dir() / "04_agriedge"
    table.to_csv(f"{out}_scores.csv", index=False)
    costs.to_csv(f"{out}_edge_cost.csv", index=False)
    print(f"\nresults written to {out}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
