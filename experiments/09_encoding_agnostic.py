"""Experiment 9 - is the leak specific to one-hot encoding?

The distributed recipe uses ``pd.get_dummies``, but published work uses other
encodings: label encoding, ordinal encoding, frequency encoding. If the leak
were an artifact of one-hot specifically, those studies would be unaffected.

It is not. Any encoding that preserves the distinction between the tokens
``'0'`` and ``'0.0'`` carries the provenance signal, because the signal *is*
that distinction. This experiment confirms it across four common encodings,
which widens the affected population from "studies that followed the Readme
verbatim" to "studies that treated these columns as categorical at all".

Run:
    python experiments/09_encoding_agnostic.py --subset ml
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from agriedge.config import (
    BINARY_LABEL,
    RANDOM_SEED,
    README_DUMMY_COLUMNS,
    results_dir,
)
from agriedge.data.curated import load_curated

#: The four columns shown in Experiment 1 to separate the classes perfectly.
LEAKING_COLUMNS: tuple[str, ...] = (
    "dns.qry.name.len",
    "mqtt.conack.flags",
    "mqtt.protoname",
    "mqtt.topic",
)


def encode_onehot(series: pd.Series) -> pd.DataFrame:
    """The recipe's own encoding: pd.get_dummies."""
    return pd.get_dummies(series, dtype=float)


def encode_label(series: pd.Series) -> pd.DataFrame:
    """Integer code per distinct token, as sklearn's LabelEncoder produces."""
    codes, _ = pd.factorize(series)
    return pd.DataFrame({"code": codes.astype(float)}, index=series.index)


def encode_ordinal_alphabetical(series: pd.Series) -> pd.DataFrame:
    """Rank of the token in sorted order - a common 'ordinal encoding'."""
    categories = sorted(series.unique())
    mapping = {token: rank for rank, token in enumerate(categories)}
    return pd.DataFrame(
        {"ordinal": series.map(mapping).astype(float)}, index=series.index
    )


def encode_frequency(series: pd.Series) -> pd.DataFrame:
    """Replace each token by its corpus frequency."""
    counts = series.value_counts(normalize=True)
    return pd.DataFrame(
        {"frequency": series.map(counts).astype(float)}, index=series.index
    )


ENCODERS = {
    "one-hot (recipe)": encode_onehot,
    "label / factorize": encode_label,
    "ordinal (alphabetical)": encode_ordinal_alphabetical,
    "frequency": encode_frequency,
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("ml", "dnn"), default="ml")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args(argv)


def score(features: pd.DataFrame, labels: pd.Series, test_size: float) -> float:
    """Held-out accuracy of a decision tree on the given encoded features."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    model.fit(x_train, y_train)
    return float(model.score(x_test, y_test))


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        frame = load_curated(args.subset, as_strings=True, nrows=args.nrows)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    labels = pd.to_numeric(frame[BINARY_LABEL]).astype(int)
    print(f"loaded {args.subset}: {len(frame):,} rows\n")
    print("Held-out accuracy of a decision tree given ONE column, by encoding:\n")

    rows: list[dict[str, object]] = []
    for column in README_DUMMY_COLUMNS:
        if column not in frame.columns:
            continue
        tokens = frame[column].astype(str)
        record: dict[str, object] = {"column": column}
        for name, encoder in ENCODERS.items():
            try:
                record[name] = score(encoder(tokens), labels, args.test_size)
            except (ValueError, MemoryError) as exc:
                print(f"  {column}/{name}: skipped ({exc})", file=sys.stderr)
                record[name] = float("nan")
        rows.append(record)

    table = pd.DataFrame(rows).set_index("column")
    pd.set_option("display.width", 200)
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    leaking = table.loc[[c for c in LEAKING_COLUMNS if c in table.index]]
    minimum = float(np.nanmin(leaking.to_numpy()))
    print(
        f"\nAcross the four leaking columns and {len(ENCODERS)} encodings, "
        f"the *minimum* single-column accuracy is {minimum:.4f}."
    )
    print(
        "The leak is a property of the token distinction, not of any "
        "particular encoding scheme."
    )

    out = results_dir() / f"09_encoding_agnostic_{args.subset}.csv"
    table.to_csv(out)
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
