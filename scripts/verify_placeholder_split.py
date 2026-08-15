#!/usr/bin/env python3
"""Check any copy of an Edge-IIoTset CSV for the placeholder-spelling artifact.

Run this against a file obtained from IEEE DataPort, Kaggle, or anywhere else,
to confirm independently that the artifact is present in the distribution and
was not introduced downstream:

    python scripts/verify_placeholder_split.py ML-EdgeIIoT-dataset.csv

Deliberately uses only the standard library and reads every field as text.
pandas' default type inference coerces the strings "0" and "0.0" to the same
float, which destroys the evidence before it can be observed -- that is the
most likely reason this went unreported. csv.reader does no such coercion.

Exit status is 1 if any audited column fully separates the classes.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

# The seven columns Readme.txt Step 5 instructs researchers to dummy-encode.
AUDIT_COLUMNS = (
    "dns.qry.name.len",
    "mqtt.conack.flags",
    "mqtt.protoname",
    "mqtt.topic",
    "http.request.method",
    "http.referer",
    "http.request.version",
)
LABEL_COLUMN = "Attack_label"
PLACEHOLDERS = ("0", "0.0")


def audit(path: str) -> int:
    csv.field_size_limit(10_000_000)

    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            print(f"error: {path} is empty", file=sys.stderr)
            return 2

        header = [h.strip().lstrip("﻿") for h in header]
        if LABEL_COLUMN not in header:
            print(f"error: no {LABEL_COLUMN!r} column in {path}", file=sys.stderr)
            print(f"       columns found: {len(header)}", file=sys.stderr)
            return 2

        label_at = header.index(LABEL_COLUMN)
        cols = {c: header.index(c) for c in AUDIT_COLUMNS if c in header}
        if not cols:
            print("error: none of the audited columns are present", file=sys.stderr)
            return 2

        # counts[column][token][label] -> rows
        counts: dict[str, dict[str, dict[str, int]]] = {
            c: defaultdict(lambda: defaultdict(int)) for c in cols
        }
        rows = 0
        for record in reader:
            if len(record) != len(header):
                continue
            label = record[label_at].strip()
            rows += 1
            for name, idx in cols.items():
                counts[name][record[idx].strip()][label] += 1

    print(f"file : {path}")
    print(f"rows : {rows:,}\n")

    leaking = []
    for name in AUDIT_COLUMNS:
        if name not in counts:
            continue
        tokens = counts[name]
        pure = sum(n for tok, labs in tokens.items()
                   for n in [sum(labs.values())] if len(labs) == 1)
        rate = pure / rows if rows else 0.0
        flag = "  <-- FULLY SEPARATES" if rate > 0.999 else ""
        print(f"{name:<24} separation_rate={rate:.4f}{flag}")

        for token in PLACEHOLDERS:
            if token in tokens:
                dist = dict(sorted(tokens[token].items()))
                shown = ", ".join(f"label {k}: {v:,}" for k, v in dist.items())
                marker = "  (single-label)" if len(dist) == 1 else ""
                print(f"    token {token!r:<6} -> {shown}{marker}")
        if rate > 0.999:
            leaking.append(name)

    print()
    if leaking:
        print(f"RESULT: {len(leaking)} column(s) recover the label on their own:")
        for name in leaking:
            print(f"  - {name}")
        print("\nIf a placeholder token maps to exactly one label, the spelling of")
        print("an ABSENT field determines the class. That is a build artifact,")
        print("not network behaviour.")
        return 1

    print("RESULT: no column fully separates the classes in this file.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(audit(sys.argv[1]))
