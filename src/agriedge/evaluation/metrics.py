"""Consistent scoring and timing for every experiment in the study.

Accuracy alone is uninformative on Edge-IIoTset, whose binary classes are
split 84.6%/15.4%. Macro-averaged F1 and balanced accuracy are reported
throughout so that a model which ignores the minority class cannot appear
competent, and per-class recall is retained because a precision-agriculture
operator cares specifically about the attacks that stop irrigation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ScoreCard:
    """Scores for one fitted model on one evaluation set."""

    model: str
    protocol: str
    split: str
    n_train: int
    n_test: int
    n_features: int
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    roc_auc: float
    fit_seconds: float
    predict_seconds: float
    per_class_recall: Mapping[str, float] = field(default_factory=dict)

    @property
    def predict_micros_per_sample(self) -> float:
        return (
            1e6 * self.predict_seconds / self.n_test if self.n_test else float("nan")
        )

    def as_row(self) -> dict[str, object]:
        """Flatten to a single row for tabulation."""
        return {
            "model": self.model,
            "protocol": self.protocol,
            "split": self.split,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "roc_auc": self.roc_auc,
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
            "predict_us_per_sample": self.predict_micros_per_sample,
        }


def _roc_auc(model: BaseEstimator, x_test, y_test) -> float:
    """ROC-AUC where the estimator supports it, else NaN."""
    if len(np.unique(y_test)) < 2:
        return float("nan")
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(x_test)
        else:
            return float("nan")
        return float(roc_auc_score(y_test, scores))
    except (ValueError, AttributeError):
        return float("nan")


def _per_class_recall(y_test, y_pred) -> dict[str, float]:
    """Recall for each observed class, keyed by class label as a string."""
    classes = np.unique(y_test)
    recalls = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    return {str(c): float(r) for c, r in zip(classes, recalls)}


def fit_and_score(
    model: BaseEstimator,
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    model_name: str,
    protocol: str,
    split: str,
) -> ScoreCard:
    """Fit a model, score it, and time both phases.

    Raises:
        ValueError: if the training set is empty or single-class.
    """
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Training and test sets must both be non-empty.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set contains a single class.")

    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(x_test)
    predict_seconds = time.perf_counter() - start

    return ScoreCard(
        model=model_name,
        protocol=protocol,
        split=split,
        n_train=int(len(x_train)),
        n_test=int(len(x_test)),
        n_features=int(x_train.shape[1]),
        accuracy=float(accuracy_score(y_test, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        precision_macro=float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        recall_macro=float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        roc_auc=_roc_auc(model, x_test, y_test),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        per_class_recall=_per_class_recall(y_test, y_pred),
    )


def to_frame(cards: tuple[ScoreCard, ...] | list[ScoreCard]) -> pd.DataFrame:
    """Tabulate score cards, most accurate first."""
    if not cards:
        return pd.DataFrame()
    return (
        pd.DataFrame([c.as_row() for c in cards])
        .sort_values(["protocol", "f1_macro"], ascending=[True, False])
        .reset_index(drop=True)
    )
