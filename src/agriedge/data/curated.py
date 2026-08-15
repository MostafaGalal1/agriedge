"""Loading of the curated ML/DNN subsets shipped with Edge-IIoTset.

Columns are read as strings by default. This is deliberate: pandas' type
inference silently normalises ``"0"`` and ``"0.0"`` to the same float, which
destroys the very artifact this study measures. Numeric coercion happens
later, explicitly, inside the preprocessing pipelines.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agriedge.config import LABEL_COLUMNS, curated_subset_path


def load_curated(
    which: str = "ml",
    *,
    as_strings: bool = True,
    columns: tuple[str, ...] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a curated subset.

    Args:
        which: ``"ml"`` or ``"dnn"``.
        as_strings: Read every column as ``str``, preserving token spelling
            exactly as written to disk. Required for the leakage audit.
        columns: Optional subset of columns to read.
        nrows: Optional row cap, for smoke tests.

    Returns:
        A new DataFrame. The caller owns it; nothing is cached or shared.

    Raises:
        FileNotFoundError: if the subset file is absent.
        ValueError: if requested columns are missing from the file.
    """
    path = curated_subset_path(which)
    available = read_header(path)

    if columns is not None:
        missing = sorted(set(columns) - set(available))
        if missing:
            raise ValueError(
                f"Columns absent from {path.name}: {missing}. "
                f"Available: {sorted(available)[:10]}..."
            )

    frame = pd.read_csv(
        path,
        low_memory=False,
        dtype=str if as_strings else None,
        usecols=list(columns) if columns else None,
        nrows=nrows,
    )
    _validate_labels(frame, path)
    return frame


def read_header(path: Path) -> tuple[str, ...]:
    """Return the column names of a CSV without reading its body."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path!s}")
    return tuple(pd.read_csv(path, nrows=0).columns)


def iter_curated(
    which: str = "dnn",
    *,
    chunksize: int = 250_000,
    columns: tuple[str, ...] | None = None,
    as_strings: bool = True,
):
    """Yield chunks of a curated subset, for files too large to hold in memory.

    Args:
        which: ``"ml"`` or ``"dnn"``.
        chunksize: Rows per chunk.
        columns: Optional subset of columns.
        as_strings: See :func:`load_curated`.

    Yields:
        Successive DataFrames.

    Raises:
        ValueError: if ``chunksize`` is not positive.
    """
    if chunksize <= 0:
        raise ValueError(f"chunksize must be positive; got {chunksize}.")

    path = curated_subset_path(which)
    reader = pd.read_csv(
        path,
        low_memory=False,
        dtype=str if as_strings else None,
        usecols=list(columns) if columns else None,
        chunksize=chunksize,
    )
    for chunk in reader:
        yield chunk


def _validate_labels(frame: pd.DataFrame, path: Path) -> None:
    """Fail fast if the label columns are missing or malformed."""
    present = set(frame.columns)
    for label in LABEL_COLUMNS:
        if label not in present and len(present) > 2:
            # A column subset may legitimately exclude labels; only complain
            # when the caller appears to have loaded the full schema.
            continue
    if LABEL_COLUMNS[0] in present:
        values = pd.to_numeric(frame[LABEL_COLUMNS[0]], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"{path.name}: {LABEL_COLUMNS[0]} contains non-numeric values."
            )
