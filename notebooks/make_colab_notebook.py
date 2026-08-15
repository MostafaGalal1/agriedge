"""Generate the Colab driver notebook for the heavy AgriEdge experiments.

Written as a generator rather than as hand-authored JSON so the cell sources
stay readable and reviewable in plain Python.

Run:
    python notebooks/make_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
CODE = "code"


def cell(kind: str, source: str) -> dict:
    """Build one notebook cell from a source string."""
    lines = source.strip("\n").splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == CODE:
        base |= {"execution_count": None, "outputs": []}
    return base


CELLS: tuple[tuple[str, str], ...] = (
    (
        MD,
        """
# AgriEdge — leakage-audited IDS benchmarking for precision agriculture

Heavy experiments for the AgriEdge study, run on Colab GPU.

**Order of operations**
1. Confirm the runtime and the signed-in account.
2. Install the `agriedge` package and dependencies.
3. Supply the data (upload the prebuilt benchmark, or rebuild from Kaggle).
4. Run the federated, deep-model, and cross-dataset experiments.

The light experiments (leakage audit, leaky-vs-clean comparison, benchmark
construction) run fine on a laptop and are already done; this notebook covers
the parts that want a GPU or a lot of RAM.
""",
    ),
    (
        MD,
        "## 1. Runtime and account check\n\n"
        "Confirm the GPU is attached **and** that the signed-in account is the "
        "intended one before anything long-running starts.",
    ),
    (
        CODE,
        """
import subprocess, sys
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "No GPU attached")
print("python:", sys.version.split()[0])

# Which Google account is this runtime attached to?
try:
    from google.colab import auth
    auth.authenticate_user()
    import requests
    info = requests.get(
        "https://www.googleapis.com/oauth2/v1/userinfo",
        headers={"Authorization": f"Bearer {auth.get_access_token()[0]}"},
        timeout=10,
    )
    print("signed in as:", info.json().get("email", "unknown"))
except Exception as exc:  # noqa: BLE001 - informational only
    print("account check unavailable:", exc)
""",
    ),
    (
        MD,
        "## 2. Install\n\n"
        "Upload `agriedge.zip` (produced by `make_bundle.sh` in the repo), then "
        "install it. Everything else comes from PyPI.",
    ),
    (
        CODE,
        """
from google.colab import files
import pathlib, zipfile

if not pathlib.Path("agriedge").exists():
    print("Upload agriedge.zip ...")
    uploaded = files.upload()
    name = next(iter(uploaded))
    with zipfile.ZipFile(name) as archive:
        archive.extractall(".")
    print("extracted:", name)

!pip install -q ./agriedge
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
import agriedge; print("agriedge", agriedge.__version__)
""",
    ),
    (
        MD,
        "## 3. Data\n\n"
        "**Option A (fast):** upload `agriedge_benchmark.parquet` (~37 MB), the "
        "benchmark already built from the raw captures.\n\n"
        "**Option B (full):** pull Edge-IIoTset from Kaggle and rebuild. Needs a "
        "`kaggle.json` API token and downloads several GB.",
    ),
    (
        CODE,
        """
# --- Option A: upload the prebuilt benchmark -------------------------------
import pathlib
from google.colab import files

BENCHMARK = pathlib.Path("agriedge_benchmark.parquet")
if not BENCHMARK.exists():
    print("Upload agriedge_benchmark.parquet ...")
    files.upload()

import pandas as pd
benchmark = pd.read_parquet(BENCHMARK)
print(f"{len(benchmark):,} rows x {benchmark.shape[1]} columns")
print(benchmark["Attack_type"].value_counts())
""",
    ),
    (
        CODE,
        """
# --- Option B: rebuild from Kaggle (skip if Option A was used) -------------
REBUILD = False  # set True to rebuild from raw captures

if REBUILD:
    from google.colab import files
    files.upload()  # kaggle.json
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    !pip install -q kaggle
    !kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
    !unzip -q -o edgeiiotset-cyber-security-dataset-of-iot-iiot.zip -d edge_iiotset

    import os
    os.environ["AGRIEDGE_DATASET_ROOT"] = "edge_iiotset/Edge-IIoTset dataset"
    from agriedge.data.agribench import BenchmarkConfig, build
    benchmark = build(BenchmarkConfig())
    benchmark.to_parquet("agriedge_benchmark.parquet", index=False)
    print(f"rebuilt: {len(benchmark):,} rows")
""",
    ),
    (
        MD,
        "## 4. Federated learning under farm heterogeneity\n\n"
        "Compares the IID split used in prior work against one client per sensor "
        "and one client per farm. The gap is the cost of heterogeneity that IID "
        "evaluation hides.",
    ),
    (
        CODE,
        """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agriedge.config import RANDOM_SEED
from agriedge.data.agribench import to_model_matrix
from agriedge.federated.fedavg import FedConfig, history_to_frame, run_fedavg
from agriedge.federated.partition import (
    partition_by_device, partition_by_farm, partition_iid,
)

SAMPLE = None  # None = use every row; the GPU can take it
frame = benchmark if SAMPLE is None else benchmark.sample(SAMPLE, random_state=RANDOM_SEED)
matrix = to_model_matrix(frame)
print("matrix:", matrix.shape)

train_idx, test_idx = train_test_split(
    range(len(matrix.features)), test_size=0.2,
    random_state=RANDOM_SEED, stratify=matrix.binary,
)
scaler = StandardScaler()
x_train = scaler.fit_transform(matrix.features.iloc[train_idx])
x_test = scaler.transform(matrix.features.iloc[test_idx])
y_train = matrix.binary.iloc[train_idx].to_numpy()
y_test = matrix.binary.iloc[test_idx].to_numpy()
train_frame = frame.iloc[train_idx].reset_index(drop=True)

FARMS = (("Soil_Moisture", "Water_Level"),
         ("Temperature_and_Humidity", "phValue"),
         ("Modbus",))

schemes = {
    "iid": partition_iid(train_frame, n_clients=5),
    "by_device": partition_by_device(train_frame),
    "by_farm": partition_by_farm(train_frame, FARMS),
}

config = FedConfig(rounds=50, local_epochs=2)
histories = []
for name, partition in schemes.items():
    print(f"\\n=== {name} ({partition.n_clients} clients) ===")
    print(partition.label_skew(matrix.binary.iloc[train_idx].reset_index(drop=True)))
    history = run_fedavg(x_train, y_train, x_test, y_test, partition, config)
    histories.append(history_to_frame(history).assign(scheme=name))

federated = pd.concat(histories, ignore_index=True)
federated.to_csv("05_federated_history.csv", index=False)
federated.groupby("scheme").tail(1)
""",
    ),
    (
        MD,
        "## 5. Communication budget\n\n"
        "A scheme that converges but will not fit a LoRaWAN uplink is not "
        "deployable. Reported against rural backhaul profiles.",
    ),
    (
        CODE,
        """
from agriedge.evaluation.edge import communication_table
from agriedge.federated.fedavg import build_model

model = build_model(x_train.shape[1], config)
n_parameters = sum(p.numel() for p in model.parameters())
print(f"model parameters: {n_parameters:,}")
communication_table(n_parameters, n_clients=5, rounds=config.rounds)
""",
    ),
    (
        MD,
        "## 6. Cross-dataset generalization\n\n"
        "Trains on AgriEdge and tests on a structurally different IIoT network "
        "without retraining. Prior work reporting 99%+ in-domain rarely does "
        "this; where it has been done, performance collapses.\n\n"
        "Set `SECOND_DATASET` to a CSV with an attack/normal label column.",
    ),
    (
        CODE,
        """
# Candidate second datasets (choose one and download into the runtime):
#   CIC-IoT-2023      https://www.unb.ca/cic/datasets/iotdataset-2023.html
#   TON_IoT           https://research.unsw.edu.au/projects/toniot-datasets
#   WUSTL-IIoT-2021   https://www.cse.wustl.edu/~jain/iiot2/
SECOND_DATASET = None  # e.g. "CICIoT2023_part1.csv"
LABEL_COLUMN = "label"

if SECOND_DATASET:
    from agriedge.evaluation.crossdomain import align_and_evaluate
    report = align_and_evaluate(
        source_matrix=matrix,
        target_csv=SECOND_DATASET,
        target_label_column=LABEL_COLUMN,
    )
    display(report)
else:
    print("Set SECOND_DATASET to run the cross-domain evaluation.")
""",
    ),
    (
        MD,
        "## 7. Export\n\nDownload every result table for the manuscript.",
    ),
    (
        CODE,
        """
import glob, zipfile
from google.colab import files

with zipfile.ZipFile("agriedge_results.zip", "w") as archive:
    for path in glob.glob("*.csv") + glob.glob("*.json"):
        archive.write(path)
files.download("agriedge_results.zip")
""",
    ),
)


def build_notebook() -> dict:
    """Assemble the notebook document."""
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [cell(kind, source) for kind, source in CELLS],
    }


def main() -> int:
    target = Path(__file__).resolve().parent / "agriedge_colab.ipynb"
    target.write_text(json.dumps(build_notebook(), indent=1), encoding="utf-8")
    print(f"wrote {target} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
