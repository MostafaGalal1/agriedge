"""Non-IID partitioning of the benchmark across simulated farm gateways.

Federated evaluations of Edge-IIoTset almost always partition rows uniformly
at random, which produces IID clients and understates the difficulty of the
real problem. A farm gateway does not see a random sample of global traffic;
it sees the devices installed on that farm. Because the AgriEdge benchmark
retains device attribution, clients can be formed the way they actually occur.

Three partitioners are provided so the paper can separate the effect of
federation from the effect of heterogeneity:

* :func:`partition_iid` - the optimistic baseline used in prior work.
* :func:`partition_by_device` - one client per sensor type; feature
  heterogeneity is maximal because each client sees a single protocol profile.
* :func:`partition_by_farm` - devices grouped into farms, which is the
  deployment the paper models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from agriedge.config import BINARY_LABEL, RANDOM_SEED
from agriedge.data.agribench import DEVICE_COLUMN


@dataclass(frozen=True)
class Partition:
    """An assignment of row indices to clients.

    Attributes:
        assignments: Client name -> row positions into the source frame.
        scheme: Human-readable name of the partitioning strategy.
    """

    assignments: Mapping[str, np.ndarray]
    scheme: str

    def __post_init__(self) -> None:
        if not self.assignments:
            raise ValueError("Partition must contain at least one client.")
        for name, index in self.assignments.items():
            if len(index) == 0:
                raise ValueError(f"Client {name!r} received no rows.")

    @property
    def n_clients(self) -> int:
        return len(self.assignments)

    @property
    def client_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.assignments))

    def sizes(self) -> dict[str, int]:
        return {name: int(len(idx)) for name, idx in sorted(self.assignments.items())}

    def label_skew(self, labels: pd.Series) -> pd.DataFrame:
        """Per-client attack rate, quantifying how non-IID the split is."""
        rows = []
        for name, index in sorted(self.assignments.items()):
            client_labels = labels.iloc[index]
            rows.append(
                {
                    "client": name,
                    "n_rows": int(len(index)),
                    "attack_rate": float(client_labels.mean()),
                    "n_normal": int((client_labels == 0).sum()),
                    "n_attack": int((client_labels == 1).sum()),
                }
            )
        return pd.DataFrame(rows)


def _validate(frame: pd.DataFrame) -> None:
    for column in (DEVICE_COLUMN, BINARY_LABEL):
        if column not in frame.columns:
            raise KeyError(
                f"Frame must carry {column!r}; build it with agribench.build()."
            )


def partition_iid(
    frame: pd.DataFrame, n_clients: int, *, seed: int = RANDOM_SEED
) -> Partition:
    """Split rows uniformly at random - the optimistic prior-work baseline.

    Raises:
        ValueError: if ``n_clients`` is not positive or exceeds the row count.
    """
    if n_clients <= 0:
        raise ValueError(f"n_clients must be positive; got {n_clients}.")
    if n_clients > len(frame):
        raise ValueError(
            f"n_clients ({n_clients}) exceeds row count ({len(frame)})."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(frame))
    chunks = np.array_split(shuffled, n_clients)
    return Partition(
        assignments={f"client_{i:02d}": chunk for i, chunk in enumerate(chunks)},
        scheme="iid",
    )


def partition_by_device(
    frame: pd.DataFrame, *, seed: int = RANDOM_SEED
) -> Partition:
    """One client per normal device; attack traffic is spread across clients.

    Attack captures carry no device attribution - an adversary is not a farm
    sensor - so attack rows are dealt out to clients at random. This models a
    campaign that reaches every farm rather than only one.
    """
    _validate(frame)
    rng = np.random.default_rng(seed)

    devices = frame[DEVICE_COLUMN]
    normal_mask = ~devices.str.startswith("attack:")
    device_names = sorted(devices[normal_mask].unique())
    if not device_names:
        raise ValueError("No normal-traffic devices found in frame.")

    assignments: dict[str, np.ndarray] = {
        name: np.flatnonzero((devices == name).to_numpy()) for name in device_names
    }

    attack_positions = np.flatnonzero(normal_mask.to_numpy() == False)  # noqa: E712
    for name, share in zip(
        device_names, np.array_split(rng.permutation(attack_positions), len(device_names))
    ):
        assignments[name] = np.concatenate([assignments[name], share])

    return Partition(assignments=assignments, scheme="by_device")


def partition_by_farm(
    frame: pd.DataFrame,
    farms: Sequence[Sequence[str]],
    *,
    seed: int = RANDOM_SEED,
) -> Partition:
    """Group devices into farms, one client per farm.

    Args:
        frame: Benchmark frame carrying device attribution.
        farms: Sequence of device-name groups, one per farm. Every group must
            be non-empty and no device may appear in two farms.
        seed: Seed governing how attack rows are dealt to farms.

    Raises:
        ValueError: if the farm specification is empty, overlapping, or names
            a device absent from the frame.
    """
    _validate(frame)
    if not farms:
        raise ValueError("At least one farm must be specified.")

    devices = frame[DEVICE_COLUMN]
    available = set(devices.unique())
    seen: set[str] = set()
    for group in farms:
        if not group:
            raise ValueError("Farm groups must be non-empty.")
        for device in group:
            if device not in available:
                raise ValueError(
                    f"Device {device!r} is not present in the frame."
                )
            if device in seen:
                raise ValueError(f"Device {device!r} assigned to two farms.")
            seen.add(device)

    rng = np.random.default_rng(seed)
    assignments: dict[str, np.ndarray] = {}
    for i, group in enumerate(farms):
        mask = devices.isin(list(group)).to_numpy()
        assignments[f"farm_{i:02d}"] = np.flatnonzero(mask)

    attack_positions = np.flatnonzero(
        devices.str.startswith("attack:").to_numpy()
    )
    names = list(assignments)
    for name, share in zip(
        names, np.array_split(rng.permutation(attack_positions), len(names))
    ):
        assignments[name] = np.concatenate([assignments[name], share])

    return Partition(assignments=assignments, scheme="by_farm")
