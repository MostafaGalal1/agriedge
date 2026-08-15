"""Canonicalisation of "absent value" placeholders.

Edge-IIoTset encodes an absent protocol field as a zero rather than as a null,
and the *spelling* of that zero differs between the normal-traffic and
attack-traffic branches of the dataset build (``"0"`` versus ``"0.0"``).
Because the distributed recipe one-hot encodes these columns, the spelling
becomes a feature, and provenance becomes linearly separable from the label.

Canonicalising every placeholder spelling to a single sentinel removes the
artifact while preserving the semantic content of the field: "this protocol
layer was not present in this packet".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: The sentinel every placeholder spelling collapses to.
ABSENT = "__absent__"

#: Spellings observed in the shipped CSVs that all mean "field not present".
DEFAULT_PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {"0", "0.0", "", " ", "nan", "NaN", "<NA>", "None", "null", "NULL"}
)


@dataclass(frozen=True)
class CanonicalisationResult:
    """Outcome of canonicalising one frame.

    Attributes:
        frame: A new frame; the input is never modified.
        replacements: Per-column count of cells rewritten to the sentinel.
        spellings_collapsed: Per-column set of distinct spellings that were
            collapsed, retained so the paper can report exactly what changed.
    """

    frame: pd.DataFrame
    replacements: dict[str, int] = field(default_factory=dict)
    spellings_collapsed: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def total_replacements(self) -> int:
        return sum(self.replacements.values())

    @property
    def affected_columns(self) -> tuple[str, ...]:
        return tuple(
            col for col, n in sorted(self.replacements.items()) if n > 0
        )

    @property
    def ambiguous_columns(self) -> tuple[str, ...]:
        """Columns where more than one placeholder spelling co-occurred.

        These are exactly the columns capable of carrying the provenance
        artifact, since a model can distinguish the spellings.
        """
        return tuple(
            col
            for col, spellings in sorted(self.spellings_collapsed.items())
            if len(spellings) > 1
        )


def canonicalise(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    tokens: frozenset[str] = DEFAULT_PLACEHOLDER_TOKENS,
    sentinel: str = ABSENT,
) -> CanonicalisationResult:
    """Collapse placeholder spellings to a single sentinel.

    Args:
        frame: Source data. Not modified.
        columns: Columns to canonicalise. Names absent from ``frame`` are
            skipped, so one column list can serve several schemas.
        tokens: Spellings treated as "absent".
        sentinel: Replacement value.

    Returns:
        A :class:`CanonicalisationResult` holding a new frame.

    Raises:
        ValueError: if ``tokens`` is empty or ``sentinel`` is one of them.
    """
    if not tokens:
        raise ValueError("tokens must contain at least one placeholder spelling.")
    if sentinel in tokens:
        raise ValueError(
            f"sentinel {sentinel!r} must not itself be a placeholder token."
        )

    updates: dict[str, pd.Series] = {}
    replacements: dict[str, int] = {}
    collapsed: dict[str, tuple[str, ...]] = {}

    for column in columns:
        if column not in frame.columns:
            continue
        rendered = frame[column].astype(str).where(frame[column].notna(), "")
        stripped = rendered.str.strip()
        mask = stripped.isin(tokens)
        count = int(mask.sum())
        replacements[column] = count
        if count:
            collapsed[column] = tuple(sorted(stripped[mask].unique()))
            updates[column] = rendered.mask(mask, sentinel)

    new_frame = frame.assign(**updates) if updates else frame.copy()
    return CanonicalisationResult(
        frame=new_frame,
        replacements=replacements,
        spellings_collapsed=collapsed,
    )
