"""Experiment 7 - leave-one-device-out across every agricultural device.

Experiment 4 withheld only Modbus, the actuation-layer device, which is
plausibly the most distinct from the perception sensors. If the collapse were
specific to Modbus it would be a curiosity rather than a finding. This sweep
withholds each of the five devices in turn.

For each held-out device the model trains on the remaining four sensors plus
80% of attack traffic, and is tested on all traffic from the held-out device
plus the remaining 20% of attacks.

Run:
    python experiments/07_lodo_sweep.py --sample 400000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from agriedge.config import AGRICULTURAL_DEVICES, RANDOM_SEED, results_dir
from agriedge.data.agribench import to_model_matrix
from agriedge.evaluation.metrics import ScoreCard, fit_and_score, to_frame
from agriedge.models.zoo import MODELS


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", default=str(results_dir() / "agriedge_benchmark.parquet")
    )
    parser.add_argument("--sample", type=int, default=400_000)
    parser.add_argument("--attack-test-fraction", type=float, default=0.2)
    return parser.parse_args(argv)


def lodo_for_device(
    matrix, holdout: str, attack_fraction: float
) -> list[ScoreCard]:
    """Train without one device's traffic; test on it."""
    is_holdout = (matrix.devices == holdout).to_numpy()
    if not is_holdout.any():
        raise ValueError(f"Device {holdout!r} absent from the benchmark.")

    is_attack = matrix.devices.str.startswith("attack:").to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    attack_to_test = rng.random(len(matrix.devices)) < attack_fraction

    test_mask = is_holdout | (is_attack & attack_to_test)
    train_mask = ~is_holdout & ~(is_attack & attack_to_test)

    x_train = matrix.features.loc[train_mask]
    y_train = matrix.binary.loc[train_mask]
    x_test = matrix.features.loc[test_mask]
    y_test = matrix.binary.loc[test_mask]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            f"Holding out {holdout!r} left a single-class split "
            f"(train classes={y_train.nunique()}, test classes={y_test.nunique()})."
        )

    print(f"  train={len(x_train):,}  test={len(x_test):,}  "
          f"test_attack_rate={y_test.mean():.4f}")

    cards: list[ScoreCard] = []
    for spec in MODELS:
        try:
            card = fit_and_score(
                spec.build(), x_train, y_train, x_test, y_test,
                model_name=spec.name, protocol="agriedge",
                split=f"lodo_{holdout}",
            )
        except (ValueError, MemoryError) as exc:
            print(f"    {spec.name}: skipped ({exc})", file=sys.stderr)
            continue
        cards.append(card)
        print(
            f"    {spec.name:22s} bal_acc={card.balanced_accuracy:.4f} "
            f"f1_macro={card.f1_macro:.4f}"
        )
    return cards


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        frame = pd.read_parquet(args.benchmark)
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.sample and args.sample < len(frame):
        frame = frame.sample(n=args.sample, random_state=RANDOM_SEED)
    matrix = to_model_matrix(frame)
    print(f"model matrix: {matrix.shape[0]:,} rows x {matrix.shape[1]} features\n")

    all_cards: list[ScoreCard] = []
    for device in AGRICULTURAL_DEVICES:
        print(f"--- holding out {device} ---")
        try:
            all_cards.extend(
                lodo_for_device(matrix, device, args.attack_test_fraction)
            )
        except ValueError as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
        print()

    if not all_cards:
        print("error: no LODO split completed.", file=sys.stderr)
        return 1

    table = to_frame(all_cards)
    pd.set_option("display.width", 250, "display.max_columns", 30)

    pivot = table.pivot_table(index="model", columns="split", values="balanced_accuracy")
    pivot["mean"] = pivot.mean(axis=1)
    pivot["worst"] = pivot.drop(columns="mean").min(axis=1)

    print("=== Balanced accuracy by held-out device ===")
    print(pivot.sort_values("mean", ascending=False).to_string(
        float_format=lambda v: f"{v:.4f}"
    ))

    f1_pivot = table.pivot_table(index="model", columns="split", values="f1_macro")
    f1_pivot["mean"] = f1_pivot.mean(axis=1)
    print("\n=== Macro-F1 by held-out device ===")
    print(f1_pivot.sort_values("mean", ascending=False).to_string(
        float_format=lambda v: f"{v:.4f}"
    ))

    out = results_dir() / "07_lodo_sweep"
    table.to_csv(f"{out}_scores.csv", index=False)
    pivot.to_csv(f"{out}_balanced_accuracy.csv")
    f1_pivot.to_csv(f"{out}_f1_macro.csv")
    print(f"\nresults written to {out}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
