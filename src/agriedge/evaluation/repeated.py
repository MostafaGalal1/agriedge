"""Repeated stratified k-fold evaluation with confidence intervals.

A dataset-centric review of this literature notes that no published study
reports k-fold cross-validation on Edge-IIoTset, relying instead on a single
80/20 split. A paper that criticises that practice must not repeat it. This
module reports every headline number as a mean over repeated stratified folds
with a 95% confidence interval, so that the effects we claim can be
distinguished from split variance.

The interval is a normal-approximation interval over fold scores. Folds within
a repeat are not independent, so it should be read as a descriptive measure of
spread rather than as a strict frequentist guarantee - the standard caveat for
cross-validation intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from agriedge.config import RANDOM_SEED

#: Metrics reported for every model. Accuracy alone is uninformative on a
#: dataset whose classes split 84.6%/15.4%.
METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy", "f1_macro")

#: 95% normal-approximation multiplier.
Z_95 = 1.959963985


@dataclass(frozen=True)
class AggregateScore:
    """Mean and dispersion of one metric for one model over folds."""

    model: str
    protocol: str
    metric: str
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_folds: int

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "protocol": self.protocol,
            "metric": self.metric,
            "mean": self.mean,
            "std": self.std,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_folds": self.n_folds,
        }

    def format(self, digits: int = 4) -> str:
        """Render as ``mean ± half-width`` for direct use in a results table."""
        return f"{self.mean:.{digits}f} ± {self.half_width:.{digits}f}"


def _score_fold(model: BaseEstimator, x_train, y_train, x_test, y_test) -> dict[str, float]:
    """Fit on one fold and return every reported metric."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "accuracy": float((predictions == np.asarray(y_test)).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
    }


def _aggregate(
    fold_scores: list[dict[str, float]], model: str, protocol: str
) -> list[AggregateScore]:
    """Summarise per-fold scores into means and intervals."""
    aggregates: list[AggregateScore] = []
    n = len(fold_scores)
    for metric in METRICS:
        values = np.asarray([s[metric] for s in fold_scores], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        margin = Z_95 * std / np.sqrt(n) if n > 1 else 0.0
        aggregates.append(
            AggregateScore(
                model=model,
                protocol=protocol,
                metric=metric,
                mean=mean,
                std=std,
                ci_low=mean - margin,
                ci_high=mean + margin,
                n_folds=n,
            )
        )
    return aggregates


def repeated_kfold(
    features: pd.DataFrame,
    labels: pd.Series,
    model_specs,
    *,
    protocol: str,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> tuple[AggregateScore, ...]:
    """Evaluate a model suite under repeated stratified k-fold.

    Args:
        features: Model matrix.
        labels: Binary labels.
        model_specs: Iterable of ``ModelSpec`` from ``agriedge.models.zoo``.
        protocol: Label recorded on every result row.
        n_splits: Folds per repeat.
        n_repeats: Number of repeats, each with a different shuffle.
        seed: Seed governing the fold structure.
        verbose: Print per-model progress.

    Raises:
        ValueError: if the inputs disagree in length, or the configuration is
            invalid for the class distribution.
    """
    if len(features) != len(labels):
        raise ValueError(
            f"features ({len(features)}) and labels ({len(labels)}) differ in length."
        )
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be >= 2 and n_repeats >= 1.")

    smallest_class = int(labels.value_counts().min())
    if smallest_class < n_splits:
        raise ValueError(
            f"Smallest class has {smallest_class} rows, fewer than n_splits="
            f"{n_splits}; stratified folding is impossible."
        )

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    folds = list(splitter.split(features, labels))

    results: list[AggregateScore] = []
    for spec in model_specs:
        fold_scores: list[dict[str, float]] = []
        for train_index, test_index in folds:
            try:
                fold_scores.append(
                    _score_fold(
                        clone(spec.build()),
                        features.iloc[train_index],
                        labels.iloc[train_index],
                        features.iloc[test_index],
                        labels.iloc[test_index],
                    )
                )
            except (ValueError, MemoryError) as exc:
                if verbose:
                    print(f"  {spec.name}: fold failed ({exc})")
                continue

        if not fold_scores:
            if verbose:
                print(f"  {spec.name}: no fold completed, skipping")
            continue

        aggregates = _aggregate(fold_scores, spec.name, protocol)
        results.extend(aggregates)
        if verbose:
            summary = {a.metric: a.format() for a in aggregates}
            print(
                f"  {spec.name:22s} acc={summary['accuracy']}  "
                f"f1_macro={summary['f1_macro']}  "
                f"bal_acc={summary['balanced_accuracy']}"
            )

    return tuple(results)


def to_frame(scores: tuple[AggregateScore, ...]) -> pd.DataFrame:
    """Tabulate aggregate scores."""
    if not scores:
        return pd.DataFrame()
    return pd.DataFrame([s.as_row() for s in scores])


def pivot_formatted(
    scores: tuple[AggregateScore, ...], metric: str = "f1_macro"
) -> pd.DataFrame:
    """Model-by-protocol table of ``mean ± half-width`` for one metric.

    Raises:
        ValueError: if ``metric`` is not one of :data:`METRICS`.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {METRICS}.")
    selected = [s for s in scores if s.metric == metric]
    if not selected:
        return pd.DataFrame()
    frame = pd.DataFrame(
        [
            {"model": s.model, "protocol": s.protocol, "value": s.format()}
            for s in selected
        ]
    )
    return frame.pivot(index="model", columns="protocol", values="value")
