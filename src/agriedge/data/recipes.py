"""The two preprocessing protocols compared in this study.

``prepare_readme`` reproduces, step for step, the recipe distributed in
``Readme.txt`` with Edge-IIoTset. It is the protocol the published literature
overwhelmingly follows, and it is leaky.

``prepare_clean`` applies the corrections this paper proposes: placeholder
canonicalisation, deduplication before splitting, structural rather than
identity encoding of high-cardinality strings, and removal of any feature that
still separates the classes perfectly after those steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from agriedge.audit.leakage import audit_columns, leaking_columns
from agriedge.config import (
    BINARY_LABEL,
    LABEL_COLUMNS,
    MULTICLASS_LABEL,
    README_DROP_COLUMNS,
    README_DUMMY_COLUMNS,
)
from agriedge.data.placeholders import ABSENT, canonicalise
from agriedge.data.textfeatures import derive_many

PROTOCOL_README = "readme"
PROTOCOL_CLEAN = "clean"


@dataclass(frozen=True)
class CleanConfig:
    """Knobs for the corrected protocol.

    Attributes:
        canonicalise_placeholders: Collapse ``"0"``/``"0.0"``/null spellings to
            one sentinel, removing the provenance artifact.
        deduplicate: Drop exact duplicate rows before any split is drawn.
        drop_separating_columns: Remove columns that still separate the classes
            perfectly after canonicalisation.
        separation_threshold: Separation rate at or above which a column is
            considered to be a relabelling of the target.
        max_onehot_cardinality: Columns with more distinct tokens than this are
            encoded structurally rather than by identity.
    """

    canonicalise_placeholders: bool = True
    deduplicate: bool = True
    drop_separating_columns: bool = True
    separation_threshold: float = 0.999
    max_onehot_cardinality: int = 16


@dataclass(frozen=True)
class PreparedData:
    """Model-ready features and labels produced by one protocol."""

    features: pd.DataFrame
    binary: pd.Series
    multiclass: pd.Series
    protocol: str
    dropped_columns: tuple[str, ...]
    notes: Mapping[str, object]

    def __post_init__(self) -> None:
        if len(self.features) != len(self.binary):
            raise ValueError(
                f"features ({len(self.features)}) and binary labels "
                f"({len(self.binary)}) differ in length."
            )
        if len(self.features) == 0:
            raise ValueError("Prepared data is empty.")

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    @property
    def n_rows(self) -> int:
        return self.features.shape[0]


def _split_labels(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract and validate both label columns."""
    for label in LABEL_COLUMNS:
        if label not in frame.columns:
            raise KeyError(f"Required label column {label!r} is missing.")

    binary = pd.to_numeric(frame[BINARY_LABEL], errors="coerce")
    if binary.isna().any():
        raise ValueError(f"{BINARY_LABEL} contains non-numeric values.")
    binary = binary.astype(int)
    if not set(binary.unique()) <= {0, 1}:
        raise ValueError(f"{BINARY_LABEL} must be binary.")

    multiclass = frame[MULTICLASS_LABEL].astype(str)
    return binary, multiclass


def _numeric_block(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce columns to float, mapping unparseable values to zero."""
    if not columns:
        return pd.DataFrame(index=frame.index)
    block = frame[columns].apply(pd.to_numeric, errors="coerce")
    return block.fillna(0.0).astype(float)


def prepare_readme(frame: pd.DataFrame) -> PreparedData:
    """Reproduce the recipe from the distributed ``Readme.txt``.

    Follows Steps 4 and 5 verbatim: drop the listed columns, drop nulls and
    duplicates, then dummy-encode the seven listed categorical columns.

    Raises:
        KeyError: if a label column is missing.
    """
    working = frame.drop(
        columns=[c for c in README_DROP_COLUMNS if c in frame.columns]
    )
    working = working.dropna(axis=0, how="any").drop_duplicates(keep="first")
    binary, multiclass = _split_labels(working)

    dummy_columns = [c for c in README_DUMMY_COLUMNS if c in working.columns]
    numeric_columns = [
        c
        for c in working.columns
        if c not in dummy_columns and c not in LABEL_COLUMNS
    ]

    encoded = (
        pd.get_dummies(working[dummy_columns], columns=dummy_columns, dtype=float)
        if dummy_columns
        else pd.DataFrame(index=working.index)
    )
    features = pd.concat([_numeric_block(working, numeric_columns), encoded], axis=1)

    return PreparedData(
        features=features,
        binary=binary.reset_index(drop=True),
        multiclass=multiclass.reset_index(drop=True),
        protocol=PROTOCOL_README,
        dropped_columns=tuple(
            c for c in README_DROP_COLUMNS if c in frame.columns
        ),
        notes={
            "rows_in": int(len(frame)),
            "rows_out": int(len(working)),
            "dummy_columns": tuple(dummy_columns),
            "n_dummy_features": int(encoded.shape[1]),
        },
    ).with_reset_features()


def prepare_clean(
    frame: pd.DataFrame, config: CleanConfig | None = None
) -> PreparedData:
    """Apply the corrected protocol proposed in this paper.

    Args:
        frame: Raw frame, read as strings so token spelling is intact.
        config: Protocol options; defaults to the paper's configuration.

    Raises:
        KeyError: if a label column is missing.
        ValueError: if every feature is removed as leaking.
    """
    settings = config or CleanConfig()

    working = frame.drop(
        columns=[c for c in README_DROP_COLUMNS if c in frame.columns]
    )

    categorical = [c for c in README_DUMMY_COLUMNS if c in working.columns]
    canonicalisation = None
    if settings.canonicalise_placeholders:
        canonicalisation = canonicalise(working, tuple(categorical))
        working = canonicalisation.frame

    if settings.deduplicate:
        before = len(working)
        working = working.drop_duplicates(keep="first")
        n_duplicates_removed = before - len(working)
    else:
        n_duplicates_removed = 0

    working = working.reset_index(drop=True)
    binary, multiclass = _split_labels(working)

    removed: list[str] = []
    if settings.drop_separating_columns and categorical:
        audited = pd.concat([working[categorical], binary.rename(BINARY_LABEL)], axis=1)
        reports = audit_columns(audited, tuple(categorical), BINARY_LABEL)
        removed = list(
            leaking_columns(reports, threshold=settings.separation_threshold)
        )
        categorical = [c for c in categorical if c not in removed]

    low_card, high_card = [], []
    for column in categorical:
        distinct = working[column].nunique(dropna=False)
        (low_card if distinct <= settings.max_onehot_cardinality else high_card).append(
            column
        )

    onehot = (
        pd.get_dummies(working[low_card], columns=low_card, dtype=float)
        if low_card
        else pd.DataFrame(index=working.index)
    )
    structural = derive_many(working, tuple(high_card))

    numeric_columns = [
        c
        for c in working.columns
        if c not in categorical and c not in removed and c not in LABEL_COLUMNS
    ]
    features = pd.concat(
        [_numeric_block(working, numeric_columns), onehot, structural], axis=1
    )
    if features.shape[1] == 0:
        raise ValueError(
            "The clean protocol removed every feature. Loosen "
            "separation_threshold or inspect the input frame."
        )

    return PreparedData(
        features=features,
        binary=binary,
        multiclass=multiclass,
        protocol=PROTOCOL_CLEAN,
        dropped_columns=tuple(
            [c for c in README_DROP_COLUMNS if c in frame.columns] + removed
        ),
        notes={
            "rows_in": int(len(frame)),
            "rows_out": int(len(working)),
            "duplicates_removed": int(n_duplicates_removed),
            "leaking_columns_removed": tuple(removed),
            "onehot_columns": tuple(low_card),
            "structural_columns": tuple(high_card),
            "placeholder_replacements": (
                canonicalisation.total_replacements if canonicalisation else 0
            ),
            "ambiguous_placeholder_columns": (
                canonicalisation.ambiguous_columns if canonicalisation else ()
            ),
            "sentinel": ABSENT,
        },
    ).with_reset_features()


def _with_reset_features(self: PreparedData) -> PreparedData:
    """Return a copy whose feature index is contiguous."""
    return PreparedData(
        features=self.features.reset_index(drop=True),
        binary=self.binary.reset_index(drop=True),
        multiclass=self.multiclass.reset_index(drop=True),
        protocol=self.protocol,
        dropped_columns=self.dropped_columns,
        notes=self.notes,
    )


PreparedData.with_reset_features = _with_reset_features  # type: ignore[attr-defined]
