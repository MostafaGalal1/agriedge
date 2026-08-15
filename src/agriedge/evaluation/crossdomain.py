"""Cross-dataset evaluation: does a detector survive a different network?

An IDS is only useful if it works on traffic it was not trained on. Prior work
on Edge-IIoTset almost always trains and tests within the same capture, which
cannot distinguish a detector from a memoriser. This module trains on one
dataset and evaluates on another without retraining.

Feature spaces rarely match across IIoT datasets, so alignment is explicit and
reported: the intersection of feature names is used, the source model is
refitted on that intersection, and the number of shared features is carried
into the result. A cross-domain score computed over three shared columns means
something very different from one computed over forty, and the reader is
entitled to know which they are looking at.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from agriedge.config import RANDOM_SEED
from agriedge.models.zoo import MODELS


@dataclass(frozen=True)
class CrossDomainResult:
    """In-domain and cross-domain scores for one model."""

    model: str
    n_shared_features: int
    n_source_test: int
    n_target: int
    in_domain_accuracy: float
    in_domain_f1_macro: float
    cross_domain_accuracy: float
    cross_domain_balanced_accuracy: float
    cross_domain_f1_macro: float
    cross_domain_attack_recall: float

    @property
    def f1_drop(self) -> float:
        """Macro-F1 lost when moving to the unseen network."""
        return self.in_domain_f1_macro - self.cross_domain_f1_macro

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "shared_features": self.n_shared_features,
            "n_source_test": self.n_source_test,
            "n_target": self.n_target,
            "in_domain_accuracy": self.in_domain_accuracy,
            "in_domain_f1_macro": self.in_domain_f1_macro,
            "cross_domain_accuracy": self.cross_domain_accuracy,
            "cross_domain_balanced_accuracy": self.cross_domain_balanced_accuracy,
            "cross_domain_f1_macro": self.cross_domain_f1_macro,
            "cross_domain_attack_recall": self.cross_domain_attack_recall,
            "f1_macro_drop": self.f1_drop,
        }


def normalise_name(name: str) -> str:
    """Reduce a feature name to a comparable form across datasets."""
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def shared_features(
    source: pd.DataFrame, target: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Return aligned column lists for the two frames.

    Matching is by normalised name. Returns source-side and target-side column
    names in a consistent order.

    Raises:
        ValueError: if the frames share no features.
    """
    source_map = {normalise_name(c): c for c in source.columns}
    target_map = {normalise_name(c): c for c in target.columns}
    common = sorted(set(source_map) & set(target_map))
    if not common:
        raise ValueError(
            "Source and target share no feature names after normalisation. "
            "Supply an explicit mapping instead."
        )
    return [source_map[k] for k in common], [target_map[k] for k in common]


def _to_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to float, mapping unparseable values to zero."""
    return frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def load_target(
    path: str | Path, label_column: str, *, attack_values: tuple[str, ...] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a target dataset and derive a binary attack label.

    Args:
        path: CSV path.
        label_column: Column carrying the class.
        attack_values: Values denoting an attack. When omitted, anything whose
            lowercase form is not in {"0", "normal", "benign"} counts as attack.

    Raises:
        FileNotFoundError: if the CSV is absent.
        KeyError: if the label column is missing.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Target dataset not found: {csv_path!s}")

    frame = pd.read_csv(csv_path, low_memory=False)
    if label_column not in frame.columns:
        raise KeyError(
            f"Label column {label_column!r} not in target; "
            f"available: {list(frame.columns)[:15]}"
        )

    raw = frame[label_column].astype(str).str.strip().str.lower()
    if attack_values is not None:
        wanted = {v.strip().lower() for v in attack_values}
        labels = raw.isin(wanted).astype(int)
    else:
        benign = {"0", "0.0", "normal", "benign", "background"}
        labels = (~raw.isin(benign)).astype(int)

    return frame.drop(columns=[label_column]), labels


def align_and_evaluate(
    source_matrix,
    target_csv: str | Path,
    target_label_column: str,
    *,
    attack_values: tuple[str, ...] | None = None,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Train on the source benchmark, evaluate on an unseen dataset.

    Args:
        source_matrix: A ``ModelMatrix`` from ``agribench.to_model_matrix``.
        target_csv: Path to the second dataset.
        target_label_column: Its label column.
        attack_values: Values in that column denoting an attack.
        test_size: In-domain held-out fraction.
        seed: Split seed.

    Returns:
        One row per model, ordered by cross-domain macro-F1.

    Raises:
        ValueError: if alignment yields no shared features.
    """
    target_features, target_labels = load_target(
        target_csv, target_label_column, attack_values=attack_values
    )
    source_columns, target_columns = shared_features(
        source_matrix.features, target_features
    )
    print(
        f"aligned on {len(source_columns)} shared features "
        f"(source has {source_matrix.features.shape[1]}, "
        f"target has {target_features.shape[1]})"
    )

    x_source = _to_numeric(source_matrix.features[source_columns])
    x_target = _to_numeric(target_features[target_columns])
    x_target.columns = source_columns  # align names so estimators accept it

    x_train, x_test, y_train, y_test = train_test_split(
        x_source,
        source_matrix.binary,
        test_size=test_size,
        random_state=seed,
        stratify=source_matrix.binary,
    )

    results: list[CrossDomainResult] = []
    for spec in MODELS:
        try:
            result = _evaluate_one(
                spec.build(),
                spec.name,
                x_train,
                y_train,
                x_test,
                y_test,
                x_target,
                target_labels,
            )
        except (ValueError, MemoryError) as exc:
            print(f"  {spec.name}: skipped ({exc})")
            continue
        results.append(result)
        print(
            f"  {spec.name:22s} in_domain_f1={result.in_domain_f1_macro:.4f} "
            f"cross_f1={result.cross_domain_f1_macro:.4f} "
            f"drop={result.f1_drop:.4f}"
        )

    if not results:
        raise ValueError("No model completed cross-domain evaluation.")
    return (
        pd.DataFrame([r.as_row() for r in results])
        .sort_values("cross_domain_f1_macro", ascending=False)
        .reset_index(drop=True)
    )


def _evaluate_one(
    model: BaseEstimator,
    name: str,
    x_train,
    y_train,
    x_test,
    y_test,
    x_target,
    y_target,
) -> CrossDomainResult:
    """Fit once, score in-domain and on the target network."""
    model.fit(x_train, y_train)
    in_domain = model.predict(x_test)
    cross = model.predict(x_target)

    attack_recall = (
        float(recall_score(y_target, cross, pos_label=1, zero_division=0))
        if 1 in set(np.unique(y_target))
        else float("nan")
    )
    return CrossDomainResult(
        model=name,
        n_shared_features=int(x_train.shape[1]),
        n_source_test=int(len(x_test)),
        n_target=int(len(x_target)),
        in_domain_accuracy=float(accuracy_score(y_test, in_domain)),
        in_domain_f1_macro=float(
            f1_score(y_test, in_domain, average="macro", zero_division=0)
        ),
        cross_domain_accuracy=float(accuracy_score(y_target, cross)),
        cross_domain_balanced_accuracy=float(
            balanced_accuracy_score(y_target, cross)
        ),
        cross_domain_f1_macro=float(
            f1_score(y_target, cross, average="macro", zero_division=0)
        ),
        cross_domain_attack_recall=attack_recall,
    )
