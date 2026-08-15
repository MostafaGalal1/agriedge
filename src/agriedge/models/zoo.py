"""The classifier suite used throughout the study.

The selection mirrors the estimators most frequently reported on Edge-IIoTset
in the published literature, so that results under the corrected protocol are
directly comparable with results under the distributed one. Every factory
returns a *fresh* estimator: no estimator instance is shared between
protocols, folds, or federated clients.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from agriedge.config import RANDOM_SEED


@dataclass(frozen=True)
class ModelSpec:
    """A named estimator factory.

    Attributes:
        name: Identifier used in result tables.
        factory: Zero-argument callable returning a new unfitted estimator.
        scales_input: Whether the estimator needs standardised features.
        edge_candidate: Whether the model is plausible for on-gateway
            inference, and therefore included in the edge-cost evaluation.
    """

    name: str
    factory: Callable[[], BaseEstimator]
    scales_input: bool
    edge_candidate: bool

    def build(self) -> BaseEstimator:
        """Return a new, unfitted estimator."""
        estimator = self.factory()
        if self.scales_input:
            return Pipeline(
                [("scale", StandardScaler()), ("clf", estimator)]
            )
        return estimator


def _decision_tree() -> BaseEstimator:
    return DecisionTreeClassifier(random_state=RANDOM_SEED)


def _random_forest() -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
    )


def _hist_gradient_boosting() -> BaseEstimator:
    return HistGradientBoostingClassifier(random_state=RANDOM_SEED)


def _logistic_regression() -> BaseEstimator:
    return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)


def _gaussian_nb() -> BaseEstimator:
    return GaussianNB()


def _mlp() -> BaseEstimator:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=60,
        random_state=RANDOM_SEED,
        early_stopping=True,
    )


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("DecisionTree", _decision_tree, scales_input=False, edge_candidate=True),
    ModelSpec("RandomForest", _random_forest, scales_input=False, edge_candidate=True),
    ModelSpec(
        "HistGradientBoosting",
        _hist_gradient_boosting,
        scales_input=False,
        edge_candidate=True,
    ),
    ModelSpec(
        "LogisticRegression",
        _logistic_regression,
        scales_input=True,
        edge_candidate=True,
    ),
    ModelSpec("GaussianNB", _gaussian_nb, scales_input=True, edge_candidate=True),
    ModelSpec("MLP", _mlp, scales_input=True, edge_candidate=False),
)


def get_model(name: str) -> ModelSpec:
    """Look up a model specification by name.

    Raises:
        ValueError: if no model with that name is registered.
    """
    for spec in MODELS:
        if spec.name == name:
            return spec
    raise ValueError(
        f"Unknown model {name!r}; registered models are "
        f"{[s.name for s in MODELS]}."
    )


def edge_models() -> tuple[ModelSpec, ...]:
    """Return the subset of models evaluated for on-gateway deployment."""
    return tuple(spec for spec in MODELS if spec.edge_candidate)
