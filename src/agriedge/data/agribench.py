"""Construction of AgriEdge, a leakage-free precision-agriculture benchmark.

The curated subsets shipped with Edge-IIoTset cannot support a domain-specific
agricultural study: Modbus is absent from them (0 of 157,800 ML rows; 150 of
2,219,201 DNN rows) and per-device identity has been stripped, leaving
``Temperature_and_Humidity`` as the only surviving MQTT topic. Both are
recoverable from the raw per-device captures, which retain full device
attribution.

Two properties are enforced during construction:

**Uniform parsing.** Every capture - normal and attack alike - is read with
identical dtype handling and identical placeholder canonicalisation. The
serialisation artifact documented in this paper arises from concatenating
differently-parsed frames, so parsing uniformly makes it structurally
impossible rather than merely correcting it after the fact.

**Preserved device attribution.** Each row carries the device or attack
capture it came from, which is what makes non-IID federated partitioning by
farm possible, and which the curated subsets discard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agriedge.config import (
    AGRICULTURAL_DEVICES,
    ATTACK_FILE_TO_TYPE,
    BINARY_LABEL,
    MULTICLASS_LABEL,
    NORMAL_CLASS,
    RANDOM_SEED,
    attack_capture_path,
    device_capture_path,
)
from agriedge.data.placeholders import DEFAULT_PLACEHOLDER_TOKENS, canonicalise

#: Provenance columns added by the builder. Excluded from features; retained
#: for federated partitioning and for auditing.
DEVICE_COLUMN = "source_device"
LAYER_COLUMN = "source_layer"
PROVENANCE_COLUMNS: tuple[str, ...] = (DEVICE_COLUMN, LAYER_COLUMN)

#: Columns that identify hosts or carry raw payload. Dropped at build time so
#: that no downstream protocol can accidentally reintroduce them.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "http.request.uri.query",
    "tcp.options",
    "tcp.payload",
    "mqtt.msg",
    "icmp.transmit_timestamp",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Sampling policy for the benchmark build.

    Attributes:
        devices: Normal-traffic devices to include.
        max_rows_per_device: Cap on sampled rows per normal device.
        max_rows_per_attack: Cap on sampled rows per attack class. Classes
            smaller than the cap are taken whole, preserving natural rarity
            for MITM and Fingerprinting.
        chunksize: Rows per read chunk.
        seed: Sampling seed.
    """

    devices: tuple[str, ...] = AGRICULTURAL_DEVICES
    max_rows_per_device: int = 150_000
    max_rows_per_attack: int = 60_000
    chunksize: int = 200_000
    seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("At least one device must be selected.")
        for field_name in ("max_rows_per_device", "max_rows_per_attack", "chunksize"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")


def count_data_rows(path: Path) -> int:
    """Count rows in a CSV excluding its header, streaming to bound memory."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path!s}")
    with open(path, "rb") as handle:
        total = sum(1 for _ in handle)
    return max(total - 1, 0)


def _sample_capture(
    path: Path, cap: int, chunksize: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Read a capture, sampling approximately ``cap`` rows uniformly.

    Sampling is applied per chunk at a constant rate, which is unbiased with
    respect to position in the file. Taking a prefix instead would bias the
    sample toward the start of the capture session.
    """
    total = count_data_rows(path)
    if total == 0:
        raise ValueError(f"Capture is empty: {path!s}")
    rate = min(1.0, cap / total)

    kept: list[pd.DataFrame] = []
    reader = pd.read_csv(path, low_memory=False, dtype=str, chunksize=chunksize)
    for chunk in reader:
        if rate >= 1.0:
            kept.append(chunk)
            continue
        mask = rng.random(len(chunk)) < rate
        if mask.any():
            kept.append(chunk.loc[mask])

    if not kept:
        raise ValueError(f"Sampling produced no rows from {path!s}.")
    return pd.concat(kept, ignore_index=True)


def _standardise(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply identical parsing and canonicalisation to every capture."""
    dropped = [c for c in IDENTIFIER_COLUMNS if c in frame.columns]
    working = frame.drop(columns=dropped)
    # Canonicalise *every* column, not only the categorical ones: any column
    # is capable of carrying the spelling artifact once it is read as text.
    result = canonicalise(
        working, tuple(working.columns), tokens=DEFAULT_PLACEHOLDER_TOKENS
    )
    return result.frame


def build(
    config: BenchmarkConfig | None = None, *, verbose: bool = True
) -> pd.DataFrame:
    """Build the AgriEdge benchmark from raw captures.

    Returns:
        A new DataFrame carrying protocol features, both label columns, and the
        provenance columns listed in :data:`PROVENANCE_COLUMNS`.

    Raises:
        FileNotFoundError: if a capture file is missing.
        ValueError: if a capture is empty or sampling yields nothing.
    """
    settings = config or BenchmarkConfig()
    rng = np.random.default_rng(settings.seed)
    parts: list[pd.DataFrame] = []

    for device in settings.devices:
        path = device_capture_path(device)
        sampled = _sample_capture(
            path, settings.max_rows_per_device, settings.chunksize, rng
        )
        standardised = _standardise(sampled)
        layer = "actuation" if device == "Modbus" else "perception"
        parts.append(
            standardised.assign(
                **{
                    BINARY_LABEL: 0,
                    MULTICLASS_LABEL: NORMAL_CLASS,
                    DEVICE_COLUMN: device,
                    LAYER_COLUMN: layer,
                }
            )
        )
        if verbose:
            print(f"  normal  {device:26s} {len(standardised):>8,} rows")

    for stem, attack_type in ATTACK_FILE_TO_TYPE.items():
        path = attack_capture_path(stem)
        sampled = _sample_capture(
            path, settings.max_rows_per_attack, settings.chunksize, rng
        )
        standardised = _standardise(sampled)
        parts.append(
            standardised.assign(
                **{
                    BINARY_LABEL: 1,
                    MULTICLASS_LABEL: attack_type,
                    DEVICE_COLUMN: f"attack:{attack_type}",
                    LAYER_COLUMN: "adversary",
                }
            )
        )
        if verbose:
            print(f"  attack  {attack_type:26s} {len(standardised):>8,} rows")

    benchmark = pd.concat(parts, ignore_index=True)
    return benchmark.sample(frac=1.0, random_state=settings.seed).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the modelling columns, excluding labels and provenance."""
    excluded = {BINARY_LABEL, MULTICLASS_LABEL, *PROVENANCE_COLUMNS}
    return tuple(c for c in frame.columns if c not in excluded)


@dataclass(frozen=True)
class ModelMatrix:
    """Encoded benchmark ready for training, with provenance kept alongside."""

    features: pd.DataFrame
    binary: pd.Series
    multiclass: pd.Series
    devices: pd.Series

    def __post_init__(self) -> None:
        lengths = {
            len(self.features),
            len(self.binary),
            len(self.multiclass),
            len(self.devices),
        }
        if len(lengths) != 1:
            raise ValueError(f"Component lengths disagree: {sorted(lengths)}.")
        if len(self.features) == 0:
            raise ValueError("Model matrix is empty.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.features.shape


def to_model_matrix(
    frame: pd.DataFrame, *, max_onehot_cardinality: int = 16
) -> ModelMatrix:
    """Encode the benchmark for modelling.

    Columns that parse cleanly as numbers become float features. Remaining
    string columns are one-hot encoded when low-cardinality, and replaced by
    structural features otherwise, so that no literal payload string is
    memorised.

    Raises:
        KeyError: if a label or provenance column is missing.
        ValueError: if ``max_onehot_cardinality`` is not positive.
    """
    if max_onehot_cardinality <= 0:
        raise ValueError("max_onehot_cardinality must be positive.")
    for column in (BINARY_LABEL, MULTICLASS_LABEL, DEVICE_COLUMN):
        if column not in frame.columns:
            raise KeyError(f"Benchmark is missing required column {column!r}.")

    from agriedge.data.textfeatures import derive_many

    candidates = feature_columns(frame)
    numeric_blocks: dict[str, pd.Series] = {}
    categorical: list[str] = []

    for column in candidates:
        converted = pd.to_numeric(frame[column], errors="coerce")
        # Treat a column as numeric only if nearly every value parses; the
        # canonicalisation sentinel is expected to fail conversion.
        parse_rate = float(converted.notna().mean())
        if parse_rate >= 0.99:
            numeric_blocks[column] = converted.fillna(0.0).astype(float)
        else:
            categorical.append(column)

    low_card = [
        c for c in categorical if frame[c].nunique(dropna=False) <= max_onehot_cardinality
    ]
    high_card = [c for c in categorical if c not in low_card]

    parts = [pd.DataFrame(numeric_blocks, index=frame.index)]
    if low_card:
        parts.append(pd.get_dummies(frame[low_card], columns=low_card, dtype=float))
    if high_card:
        parts.append(derive_many(frame, tuple(high_card)))

    features = pd.concat(parts, axis=1).reset_index(drop=True)
    if features.shape[1] == 0:
        raise ValueError("Encoding produced no features.")

    return ModelMatrix(
        features=features,
        binary=pd.to_numeric(frame[BINARY_LABEL]).astype(int).reset_index(drop=True),
        multiclass=frame[MULTICLASS_LABEL].astype(str).reset_index(drop=True),
        devices=frame[DEVICE_COLUMN].astype(str).reset_index(drop=True),
    )
