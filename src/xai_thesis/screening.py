"""Constant-column and near-duplicate screen over a cleaned dataset's numeric features.

A column is removed by the screen if it is constant, or if it repeats a retained column's value
in all but a negligible share of rows (max_differing_share). Pearson correlation only nominates
the candidate pairs (|r| >= min_abs_correlation): two columns with the same values in nearly every
row always correlate near 1, but the reverse does not hold - CIC-IDS2017's Flow Duration and Fwd
IAT Total correlate at r = 0.9985 yet differ in ~77% of rows, so they are distinct quantities and
both are kept. The record-by-record comparison is what decides.

check_screen compares the screen's result with a dataset config's screen_drop_columns. Which
member of a duplicate pair to drop is a judgement call (the more interpretable name is kept), so
it checks that every flagged pair loses exactly one member rather than which one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

SCREEN_COLUMNS = ["check", "column", "other_column", "r", "n_differing", "share_differing", "flagged"]


def screen_features(
    features: pd.DataFrame,
    *,
    min_abs_correlation: float = 0.99,
    max_differing_share: float = 1e-4,
) -> pd.DataFrame:
    """One row per constant column and per candidate pair, with the evidence for each.

    check is "constant" or "pair"; for a pair, r is the Pearson correlation and n_differing the
    number of rows whose two values differ (NaN equals NaN). flagged marks a constant column, or a
    pair differing in at most max_differing_share of rows. Non-numeric columns are ignored.
    """
    numeric = features.select_dtypes(include="number")
    n_rows = len(numeric)
    constant = [column for column in numeric.columns if numeric[column].nunique(dropna=True) <= 1]
    rows = [
        {"check": "constant", "column": column, "other_column": None, "r": np.nan,
         "n_differing": 0, "share_differing": 0.0, "flagged": True}
        for column in constant
    ]

    varying = numeric.drop(columns=constant)
    correlation = varying.corr().abs().to_numpy()
    columns = list(varying.columns)
    for i, j in zip(*np.triu_indices(len(columns), k=1)):
        if not correlation[i, j] >= min_abs_correlation:
            continue
        a, b = varying[columns[i]].to_numpy(dtype=float), varying[columns[j]].to_numpy(dtype=float)
        n_differing = int((~np.isclose(a, b, rtol=1e-9, atol=0.0, equal_nan=True)).sum())
        share = n_differing / n_rows if n_rows else 0.0
        rows.append(
            {"check": "pair", "column": columns[i], "other_column": columns[j], "r": float(correlation[i, j]),
             "n_differing": n_differing, "share_differing": share, "flagged": share <= max_differing_share}
        )
    screen = pd.DataFrame(rows, columns=SCREEN_COLUMNS)
    return screen.sort_values(["check", "share_differing", "r"], ascending=[True, True, False], ignore_index=True)


def check_screen(screen: pd.DataFrame, screen_drop_columns: Sequence[str]) -> list[str]:
    """Inconsistencies between the screen and a config's screen_drop_columns; empty if none.

    Every constant column must be dropped; every flagged pair must keep at least one member and
    drop at least one; every dropped column must be constant or a flagged pair's dropped member.
    """
    dropped = set(screen_drop_columns)
    problems: list[str] = []
    justified: set[str] = set()

    for row in screen.itertuples(index=False):
        if row.check == "constant":
            justified.add(row.column)
            if row.column not in dropped:
                problems.append(f"constant column {row.column!r} is not in screen_drop_columns")
        elif row.flagged:
            pair = {row.column, row.other_column}
            if not pair & dropped:
                problems.append(
                    f"{row.column!r} and {row.other_column!r} differ in {row.n_differing} rows "
                    "but neither is in screen_drop_columns"
                )
            elif pair <= dropped:
                problems.append(f"both {row.column!r} and {row.other_column!r} are in screen_drop_columns")
            else:
                justified |= pair & dropped

    for column in sorted(dropped - justified):
        problems.append(f"screen_drop_columns lists {column!r}, which the screen does not flag")
    return problems
