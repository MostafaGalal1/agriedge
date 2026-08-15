"""Experiment 8 - does the artifact inflate deep models too?

The headline Edge-IIoTset results come from deep architectures, not from the
classical estimators of Experiment 2. A reviewer may reasonably ask whether a
deep model would have been robust to the serialisation artifact. This
experiment answers it directly, training an MLP and a 1D-CNN under both
protocols on identical splits.

Runs on CUDA, Apple MPS, or CPU, whichever is available.

Run:
    python experiments/08_deep_baselines.py --subset ml --epochs 15
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agriedge.config import RANDOM_SEED, results_dir
from agriedge.data.curated import load_curated
from agriedge.data.recipes import CleanConfig, prepare_clean, prepare_readme
from agriedge.models.deep import (
    TORCH_AVAILABLE,
    DeepConfig,
    DeepResult,
    to_frame,
    train_and_score,
)

ARCHITECTURES: tuple[str, ...] = ("mlp", "cnn1d")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("ml", "dnn"), default="ml")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args(argv)


def evaluate(prepared, config: DeepConfig, test_size: float) -> list[DeepResult]:
    """Train every architecture on one prepared dataset."""
    x_train, x_test, y_train, y_test = train_test_split(
        prepared.features,
        prepared.binary,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=prepared.binary,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    results: list[DeepResult] = []
    for architecture in ARCHITECTURES:
        print(f"  {architecture}:")
        try:
            result = train_and_score(
                architecture,
                x_train_scaled,
                y_train.to_numpy(),
                x_test_scaled,
                y_test.to_numpy(),
                config,
                protocol=prepared.protocol,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"    skipped ({exc})", file=sys.stderr)
            continue
        results.append(result)
        print(
            f"    acc={result.accuracy:.4f} f1_macro={result.f1_macro:.4f} "
            f"bal_acc={result.balanced_accuracy:.4f} "
            f"({result.n_parameters:,} params, {result.train_seconds:.1f}s "
            f"on {result.device})"
        )
    return results


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not TORCH_AVAILABLE:
        print(
            "error: PyTorch is not installed. Install with:\n"
            "  pip install 'agriedge[torch]'",
            file=sys.stderr,
        )
        return 1

    try:
        frame = load_curated(args.subset, as_strings=True, nrows=args.nrows)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"loaded {args.subset}: {len(frame):,} rows\n")
    config = DeepConfig(epochs=args.epochs, batch_size=args.batch_size)

    print("--- Protocol A: recipe as distributed ---")
    leaky = prepare_readme(frame)
    print(f"  rows={leaky.n_rows:,} features={leaky.n_features}")
    leaky_results = evaluate(leaky, config, args.test_size)

    print("\n--- Protocol B: corrected protocol ---")
    clean = prepare_clean(frame, CleanConfig())
    print(f"  rows={clean.n_rows:,} features={clean.n_features}")
    clean_results = evaluate(clean, config, args.test_size)

    table = to_frame(leaky_results + clean_results)
    if table.empty:
        print("error: no deep model completed.", file=sys.stderr)
        return 1

    pd.set_option("display.width", 220, "display.max_columns", 20)
    pivot = table.pivot_table(
        index="model", columns="protocol", values=["accuracy", "f1_macro"]
    )
    if "readme" in table["protocol"].values and "clean" in table["protocol"].values:
        pivot[("delta", "accuracy")] = (
            pivot[("accuracy", "readme")] - pivot[("accuracy", "clean")]
        )
        pivot[("delta", "f1_macro")] = (
            pivot[("f1_macro", "readme")] - pivot[("f1_macro", "clean")]
        )

    print("\n=== Deep baselines: leaky vs clean ===")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    out = results_dir() / f"08_deep_baselines_{args.subset}"
    table.to_csv(f"{out}.csv", index=False)
    print(f"\nresults written to {out}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
