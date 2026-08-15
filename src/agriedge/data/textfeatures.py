"""Generalisable features for high-cardinality string columns.

One-hot encoding a column such as ``http.request.version`` - whose attack-side
tokens include whole injected request lines - lets a model memorise literal
payload strings. Memorised strings do not transfer to a different network, and
they are trivially evaded by changing a byte.

These functions replace token identity with structural properties of the
string (length, character-class composition, entropy, presence of syntax
associated with injection). Such features describe *how a value is malformed*
rather than *which malformed value it is*, so they remain meaningful on
traffic the model has never seen.
"""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd

#: Characters whose presence in a protocol field is structurally suspicious.
INJECTION_MARKERS: tuple[tuple[str, str], ...] = (
    ("has_angle_bracket", "<>"),
    ("has_quote", "'\""),
    ("has_path_separator", "/\\"),
    ("has_parenthesis", "()"),
    ("has_semicolon", ";"),
    ("has_equals", "="),
    ("has_dollar", "$"),
    ("has_whitespace", " \t"),
)

FEATURE_SUFFIXES: tuple[str, ...] = (
    "length",
    "digit_ratio",
    "alpha_ratio",
    "punct_ratio",
    "entropy",
    *(name for name, _ in INJECTION_MARKERS),
)


def shannon_entropy(value: str) -> float:
    """Return the Shannon entropy of a string in bits per character."""
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum(
        (n / total) * math.log2(n / total) for n in counts.values()
    )


def derive(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Expand one string column into structural numeric features.

    Args:
        series: Column of strings. Nulls are treated as empty strings.
        prefix: Prefix for the generated column names.

    Returns:
        A new DataFrame of float/bool features indexed like ``series``.

    Raises:
        ValueError: if ``prefix`` is empty.
    """
    if not prefix:
        raise ValueError("prefix must be a non-empty string.")

    text = series.astype(str).where(series.notna(), "")
    length = text.str.len()
    safe_length = length.replace(0, pd.NA)

    features = {
        f"{prefix}__length": length.astype(float),
        f"{prefix}__digit_ratio": (
            text.str.count(r"\d") / safe_length
        ).fillna(0.0).astype(float),
        f"{prefix}__alpha_ratio": (
            text.str.count(r"[A-Za-z]") / safe_length
        ).fillna(0.0).astype(float),
        f"{prefix}__punct_ratio": (
            text.str.count(r"[^\w\s]") / safe_length
        ).fillna(0.0).astype(float),
        f"{prefix}__entropy": text.map(shannon_entropy).astype(float),
    }
    for name, charset in INJECTION_MARKERS:
        pattern = "[" + "".join(f"\\{c}" for c in charset) + "]"
        features[f"{prefix}__{name}"] = (
            text.str.contains(pattern, regex=True, na=False).astype(float)
        )

    return pd.DataFrame(features, index=series.index)


def derive_many(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Apply :func:`derive` to several columns and concatenate the results."""
    parts = [
        derive(frame[column], prefix=column)
        for column in columns
        if column in frame.columns
    ]
    if not parts:
        return pd.DataFrame(index=frame.index)
    return pd.concat(parts, axis=1)
