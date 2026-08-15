"""Experiment 5 - federated detection under realistic farm heterogeneity.

Compares FedAvg under three client constructions: the IID split used in prior
work, one client per sensor type, and one client per farm. The gap between IID
and the device/farm splits is the cost of heterogeneity that IID evaluations
hide. Communication cost is reported against rural backhaul budgets, because
a scheme that converges but cannot fit the uplink is not deployable.

Run:
    python experiments/05_federated.py --sample 300000 --rounds 20
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agriedge.config import RANDOM_SEED, results_dir
from agriedge.data.agribench import to_model_matrix
from agriedge.evaluation.edge import communication_table
from agriedge.federated.fedavg import (
    TORCH_AVAILABLE,
    FedConfig,
    history_to_frame,
    run_fedavg,
)
from agriedge.federated.partition import (
    partition_by_device,
    partition_by_farm,
    partition_iid,
)

#: Farms modelled in the paper: a soil-and-water farm, a climate-and-pH farm,
#: and an irrigation-control site fronted by the Modbus gateway.
DEFAULT_FARMS: tuple[tuple[str, ...], ...] = (
    ("Soil_Moisture", "Water_Level"),
    ("Temperature_and_Humidity", "phValue"),
    ("Modbus",),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default=str(results_dir() / "agriedge_benchmark.parquet"),
    )
    parser.add_argument("--sample", type=int, default=300_000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args(argv)


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
        frame = pd.read_parquet(args.benchmark)
    except (FileNotFoundError, OSError) as exc:
        print(f"error: could not read benchmark: {exc}", file=sys.stderr)
        return 1

    if args.sample and args.sample < len(frame):
        frame = frame.sample(n=args.sample, random_state=RANDOM_SEED)
    matrix = to_model_matrix(frame)
    print(f"model matrix: {matrix.shape[0]:,} rows x {matrix.shape[1]} features")

    train_idx, test_idx = train_test_split(
        range(len(matrix.features)),
        test_size=args.test_size,
        random_state=RANDOM_SEED,
        stratify=matrix.binary,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(matrix.features.iloc[train_idx])
    x_test = scaler.transform(matrix.features.iloc[test_idx])
    y_train = matrix.binary.iloc[train_idx].to_numpy()
    y_test = matrix.binary.iloc[test_idx].to_numpy()

    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    config = FedConfig(rounds=args.rounds, local_epochs=args.local_epochs)

    schemes = {
        "iid": partition_iid(train_frame, n_clients=5),
        "by_device": partition_by_device(train_frame),
        "by_farm": partition_by_farm(train_frame, DEFAULT_FARMS),
    }

    all_history = []
    for name, partition in schemes.items():
        print(f"\n--- {name} ({partition.n_clients} clients) ---")
        skew = partition.label_skew(matrix.binary.iloc[train_idx].reset_index(drop=True))
        print(skew.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        try:
            history = run_fedavg(
                x_train, y_train, x_test, y_test, partition, config
            )
        except (RuntimeError, ValueError) as exc:
            print(f"error in {name}: {exc}", file=sys.stderr)
            continue
        table = history_to_frame(history).assign(scheme=name)
        all_history.append(table)

    if not all_history:
        print("error: no federated run completed.", file=sys.stderr)
        return 1

    combined = pd.concat(all_history, ignore_index=True)
    final = combined.groupby("scheme").tail(1).set_index("scheme")
    print("\n=== Final global model by client construction ===")
    print(
        final[["accuracy", "balanced_accuracy", "f1_macro"]].to_string(
            float_format=lambda v: f"{v:.4f}"
        )
    )

    n_parameters = sum(
        p.numel()
        for p in __import__("agriedge.federated.fedavg", fromlist=["build_model"])
        .build_model(x_train.shape[1], config)
        .parameters()
    )
    print(f"\n=== Communication cost (model has {n_parameters:,} parameters) ===")
    comms = communication_table(n_parameters, n_clients=5, rounds=args.rounds)
    print(comms.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = results_dir() / "05_federated"
    combined.to_csv(f"{out}_history.csv", index=False)
    comms.to_csv(f"{out}_communication.csv", index=False)
    print(f"\nresults written to {out}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
