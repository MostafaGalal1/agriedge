"""Deployment cost of a trained detector on a farm gateway.

Accuracy is not the binding constraint in precision agriculture. A gateway in
a field is typically a single-board computer sharing power with the irrigation
controller and reaching the internet over a cellular or LoRa backhaul, so the
questions that decide deployability are how large the model is, how long one
inference takes, and how much traffic federated training would add to a link
that is already the scarcest resource on the farm.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

#: Uplink budgets representative of rural agricultural backhaul, in kbit/s.
BACKHAUL_PROFILES: dict[str, float] = {
    "lorawan_sf7": 5.47,
    "nb_iot": 250.0,
    "lte_cat_m1": 1000.0,
    "rural_4g": 5000.0,
}


@dataclass(frozen=True)
class EdgeCost:
    """Deployment footprint of one fitted model."""

    model: str
    serialized_bytes: int
    n_parameters: int
    median_latency_us: float
    p95_latency_us: float
    throughput_per_second: float
    batch_size: int

    @property
    def serialized_kib(self) -> float:
        return self.serialized_bytes / 1024.0

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "serialized_kib": self.serialized_kib,
            "n_parameters": self.n_parameters,
            "median_latency_us": self.median_latency_us,
            "p95_latency_us": self.p95_latency_us,
            "throughput_per_second": self.throughput_per_second,
            "batch_size": self.batch_size,
        }


def _count_parameters(model: BaseEstimator) -> int:
    """Best-effort parameter count across estimator families."""
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        intercept = np.asarray(getattr(model, "intercept_", []))
        return int(coef.size + intercept.size)
    if hasattr(model, "tree_"):
        return int(model.tree_.node_count)
    if hasattr(model, "estimators_"):
        estimators = np.ravel(model.estimators_)
        return int(
            sum(
                e.tree_.node_count
                for e in estimators
                if hasattr(e, "tree_")
            )
        )
    if hasattr(model, "theta_"):
        return int(np.asarray(model.theta_).size * 2)
    return -1


def _unwrap(model: BaseEstimator) -> BaseEstimator:
    """Return the final estimator of a pipeline, or the model itself."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


def measure(
    model: BaseEstimator,
    x_sample: np.ndarray,
    *,
    model_name: str,
    repeats: int = 30,
    batch_size: int = 1,
) -> EdgeCost:
    """Measure serialised size and inference latency of a fitted model.

    Latency is reported as a median and a 95th percentile over ``repeats``
    timed batches. The tail matters more than the mean for an irrigation
    controller, where a late verdict is a missed actuation window.

    Raises:
        ValueError: if the sample is empty or the parameters are invalid.
    """
    if len(x_sample) == 0:
        raise ValueError("x_sample must be non-empty.")
    if repeats <= 0 or batch_size <= 0:
        raise ValueError("repeats and batch_size must be positive.")

    payload = pickle.dumps(model)
    batch = np.asarray(x_sample)[:batch_size]

    model.predict(batch)  # warm caches so the first timed call is representative

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(batch)
        timings.append((time.perf_counter() - start) * 1e6)

    latencies = np.asarray(timings)
    median = float(np.median(latencies))
    return EdgeCost(
        model=model_name,
        serialized_bytes=len(payload),
        n_parameters=_count_parameters(_unwrap(model)),
        median_latency_us=median,
        p95_latency_us=float(np.percentile(latencies, 95)),
        throughput_per_second=(1e6 * batch_size / median) if median > 0 else float("inf"),
        batch_size=batch_size,
    )


def round_trip_seconds(
    parameter_count: int,
    n_clients: int,
    *,
    bits_per_parameter: int = 32,
    uplink_kbps: float,
) -> float:
    """Seconds of uplink needed for one federated round.

    Assumes each selected client uploads a full dense parameter vector.

    Raises:
        ValueError: if any argument is non-positive.
    """
    if parameter_count <= 0 or n_clients <= 0 or uplink_kbps <= 0:
        raise ValueError("All arguments must be positive.")
    bits = parameter_count * bits_per_parameter * n_clients
    return bits / (uplink_kbps * 1000.0)


def communication_table(
    parameter_count: int, n_clients: int, rounds: int
) -> pd.DataFrame:
    """Uplink time per round and for full training, across backhaul profiles."""
    if rounds <= 0:
        raise ValueError("rounds must be positive.")
    rows = []
    for profile, kbps in BACKHAUL_PROFILES.items():
        per_round = round_trip_seconds(
            parameter_count, n_clients, uplink_kbps=kbps
        )
        rows.append(
            {
                "backhaul": profile,
                "uplink_kbps": kbps,
                "seconds_per_round": per_round,
                "minutes_full_training": per_round * rounds / 60.0,
            }
        )
    return pd.DataFrame(rows)


def to_frame(costs: tuple[EdgeCost, ...] | list[EdgeCost]) -> pd.DataFrame:
    """Tabulate edge costs, cheapest first."""
    if not costs:
        return pd.DataFrame()
    return (
        pd.DataFrame([c.as_row() for c in costs])
        .sort_values("median_latency_us")
        .reset_index(drop=True)
    )
