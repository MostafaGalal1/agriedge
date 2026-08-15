"""Detection and quantification of label leakage in categorical columns.

The audit answers one question per column: *how much of a classifier's
apparent skill can this column supply on its own, without any network
behaviour being modelled?*

Three independent measures are computed so that no single statistic carries
the argument:

1. **Token purity** - tokens that occur under exactly one label. A column whose
   tokens are all pure is a relabelling of the target.
2. **Single-column test accuracy** - a decision tree fitted to that column
   alone, scored on a held-out stratified split.
3. **Normalised mutual information** with the label, which is scale-free and
   does not depend on any estimator.

A fourth, mechanism-specific probe (``zero_spelling_report``) tests the
hypothesis that the placeholder token for "absent" was serialised differently
in the normal-traffic and attack-traffic branches of the dataset build, which
would make provenance - not behaviour - linearly separable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from agriedge.config import RANDOM_SEED

# Placeholder spellings produced by the two serialisation branches. A column
# containing both is evidence that two differently-parsed frames were
# concatenated without normalising missing values.
STRING_ZERO = "0"
FLOAT_ZERO = "0.0"

MISSING_TOKEN = "<NA>"


@dataclass(frozen=True)
class ZeroSpellingReport:
    """Class distribution of the two spellings of the zero placeholder."""

    column: str
    string_zero_normal: int
    string_zero_attack: int
    float_zero_normal: int
    float_zero_attack: int

    @property
    def is_provenance_marker(self) -> bool:
        """True when each spelling occurs under exactly one label.

        This is the signature of a concatenation artifact: the spelling of an
        *absent* value, which carries no network semantics, determines the
        label with certainty.
        """
        string_pure = (self.string_zero_normal == 0) != (self.string_zero_attack == 0)
        float_pure = (self.float_zero_normal == 0) != (self.float_zero_attack == 0)
        both_present = (
            self.string_zero_normal + self.string_zero_attack > 0
            and self.float_zero_normal + self.float_zero_attack > 0
        )
        return both_present and string_pure and float_pure

    @property
    def rows_covered(self) -> int:
        return (
            self.string_zero_normal
            + self.string_zero_attack
            + self.float_zero_normal
            + self.float_zero_attack
        )


@dataclass(frozen=True)
class ColumnLeakageReport:
    """Leakage statistics for a single categorical column."""

    column: str
    n_rows: int
    n_tokens: int
    pure_normal_tokens: int
    pure_attack_tokens: int
    rows_separated: int
    single_column_accuracy: float
    normalized_mutual_info: float
    zero_spelling: ZeroSpellingReport | None

    @property
    def separation_rate(self) -> float:
        """Fraction of rows lying under a token unique to one label."""
        return self.rows_separated / self.n_rows if self.n_rows else 0.0

    @property
    def is_fully_separating(self) -> bool:
        """True when every row is covered by a label-pure token."""
        return self.n_rows > 0 and self.rows_separated == self.n_rows


def _as_tokens(series: pd.Series) -> pd.Series:
    """Render a column as strings exactly as ``pd.get_dummies`` would see it."""
    return series.astype(str).where(series.notna(), MISSING_TOKEN)


def zero_spelling_report(
    tokens: pd.Series, labels: pd.Series, column: str
) -> ZeroSpellingReport | None:
    """Compare label distributions of ``"0"`` against ``"0.0"``.

    Returns ``None`` when the column does not contain both spellings, in which
    case the provenance hypothesis is not testable for that column.
    """
    is_string_zero = tokens == STRING_ZERO
    is_float_zero = tokens == FLOAT_ZERO
    if not (is_string_zero.any() and is_float_zero.any()):
        return None

    attack = labels == 1
    return ZeroSpellingReport(
        column=column,
        string_zero_normal=int((is_string_zero & ~attack).sum()),
        string_zero_attack=int((is_string_zero & attack).sum()),
        float_zero_normal=int((is_float_zero & ~attack).sum()),
        float_zero_attack=int((is_float_zero & attack).sum()),
    )


def _single_column_accuracy(
    tokens: pd.Series, labels: pd.Series, test_size: float, seed: int
) -> float:
    """Held-out accuracy of a decision tree fitted to this column alone."""
    if labels.nunique() < 2:
        return float("nan")

    x_train, x_test, y_train, y_test = train_test_split(
        tokens.to_frame(),
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    model = DecisionTreeClassifier(random_state=seed)
    model.fit(encoder.fit_transform(x_train), y_train)
    return float(model.score(encoder.transform(x_test), y_test))


def audit_column(
    frame: pd.DataFrame,
    column: str,
    label_column: str,
    *,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> ColumnLeakageReport:
    """Quantify how much label information ``column`` carries on its own.

    Args:
        frame: Source data, unmodified.
        column: Categorical column to audit.
        label_column: Binary label column (0 = normal, 1 = attack).
        test_size: Held-out fraction for the single-column classifier.
        seed: Random seed for the split and the tree.

    Raises:
        KeyError: if ``column`` or ``label_column`` is absent.
        ValueError: if the label column is not binary-valued.
    """
    for required in (column, label_column):
        if required not in frame.columns:
            raise KeyError(f"Column {required!r} not present in frame.")

    labels = pd.to_numeric(frame[label_column], errors="coerce")
    if labels.isna().any():
        raise ValueError(
            f"Label column {label_column!r} contains non-numeric values."
        )
    labels = labels.astype(int)
    observed = set(labels.unique())
    if not observed <= {0, 1}:
        raise ValueError(
            f"Label column {label_column!r} must be binary; saw {sorted(observed)}."
        )

    tokens = _as_tokens(frame[column])
    counts = pd.crosstab(tokens, labels)
    for missing_label in (0, 1):
        if missing_label not in counts.columns:
            counts[missing_label] = 0

    normal_only = counts[(counts[1] == 0) & (counts[0] > 0)]
    attack_only = counts[(counts[0] == 0) & (counts[1] > 0)]
    rows_separated = int(normal_only[0].sum() + attack_only[1].sum())

    return ColumnLeakageReport(
        column=column,
        n_rows=len(frame),
        n_tokens=int(counts.shape[0]),
        pure_normal_tokens=int(normal_only.shape[0]),
        pure_attack_tokens=int(attack_only.shape[0]),
        rows_separated=rows_separated,
        single_column_accuracy=_single_column_accuracy(
            tokens, labels, test_size, seed
        ),
        normalized_mutual_info=float(
            normalized_mutual_info_score(tokens.to_numpy(), labels.to_numpy())
        ),
        zero_spelling=zero_spelling_report(tokens, labels, column),
    )


def audit_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label_column: str,
    *,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[ColumnLeakageReport, ...]:
    """Audit several columns, returning reports in the order given.

    Columns absent from ``frame`` are skipped rather than raising, so the same
    column list can be applied to subsets with differing schemas.
    """
    return tuple(
        audit_column(
            frame, column, label_column, test_size=test_size, seed=seed
        )
        for column in columns
        if column in frame.columns
    )


def leaking_columns(
    reports: tuple[ColumnLeakageReport, ...], *, threshold: float = 1.0
) -> tuple[str, ...]:
    """Names of columns whose separation rate meets ``threshold``."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1]; got {threshold}.")
    return tuple(r.column for r in reports if r.separation_rate >= threshold)


def duplicate_statistics(
    frame: pd.DataFrame, label_columns: tuple[str, ...]
) -> dict[str, float]:
    """Duplicate-row rates, which inflate accuracy under random splitting.

    Exact duplicates spanning a train/test boundary are memorised rather than
    generalised, so their rate bounds the optimism of any random split.
    """
    feature_columns = [c for c in frame.columns if c not in label_columns]
    if not feature_columns:
        raise ValueError("Frame contains no feature columns.")

    n = len(frame)
    exact = int(frame.duplicated().sum())
    feature_only = int(frame.duplicated(subset=feature_columns).sum())
    contradictory = feature_only - exact
    return {
        "n_rows": float(n),
        "exact_duplicate_rows": float(exact),
        "exact_duplicate_rate": exact / n if n else 0.0,
        "feature_duplicate_rows": float(feature_only),
        "feature_duplicate_rate": feature_only / n if n else 0.0,
        "contradictory_duplicate_rows": float(contradictory),
    }


def summarize(reports: tuple[ColumnLeakageReport, ...]) -> pd.DataFrame:
    """Return a tidy table of the audit, sorted by severity."""
    if not reports:
        return pd.DataFrame()
    rows = [
        {
            "column": r.column,
            "tokens": r.n_tokens,
            "pure_normal_tokens": r.pure_normal_tokens,
            "pure_attack_tokens": r.pure_attack_tokens,
            "rows_separated": r.rows_separated,
            "separation_rate": r.separation_rate,
            "single_column_accuracy": r.single_column_accuracy,
            "nmi_with_label": r.normalized_mutual_info,
            "provenance_marker": bool(
                r.zero_spelling and r.zero_spelling.is_provenance_marker
            ),
        }
        for r in reports
    ]
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["separation_rate", "single_column_accuracy"], ascending=False
    ).reset_index(drop=True)


def np_seed_state() -> np.random.Generator:
    """Return a seeded generator for any stochastic step in the audit."""
    return np.random.default_rng(RANDOM_SEED)
