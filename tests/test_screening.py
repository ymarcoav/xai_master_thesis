import numpy as np
import pandas as pd

from xai_thesis.screening import check_screen, screen_features


def _features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = rng.normal(size=10_000)
    alias = base.copy()
    alias[0] += 0.5  # differs in 1 of 10,000 rows
    correlated = base + rng.normal(scale=0.01, size=10_000)  # r ~ 0.9999, but no value equal
    return pd.DataFrame(
        {
            "base": base,
            "alias": alias,
            "correlated": correlated,
            "constant": np.zeros(10_000),
            "independent": rng.normal(size=10_000),
            "category": ["x"] * 10_000,
        }
    )


def test_screen_flags_constant_columns_and_value_identical_pairs_but_not_merely_correlated_ones() -> None:
    screen = screen_features(_features(), max_differing_share=1e-4)
    by_pair = {frozenset((row.column, row.other_column)): row for row in screen.itertuples() if row.check == "pair"}

    assert screen.loc[screen["check"] == "constant", "column"].tolist() == ["constant"]
    assert by_pair[frozenset(("base", "alias"))].n_differing == 1
    assert by_pair[frozenset(("base", "alias"))].flagged
    # Correlated far above 0.99 but different in every row: a candidate, not a duplicate.
    assert by_pair[frozenset(("base", "correlated"))].r > 0.999
    assert by_pair[frozenset(("base", "correlated"))].n_differing == 10_000
    assert not by_pair[frozenset(("base", "correlated"))].flagged
    # Uncorrelated and non-numeric columns are never candidates.
    assert "independent" not in set(screen["column"]) | set(screen["other_column"].dropna())
    assert "category" not in set(screen["column"])


def test_check_screen_accepts_a_matching_config_and_names_each_mismatch() -> None:
    screen = screen_features(_features())

    assert check_screen(screen, ["constant", "alias"]) == []
    # Either member of a duplicate pair may be the one dropped.
    assert check_screen(screen, ["constant", "base"]) == []

    problems = check_screen(screen, ["base", "alias", "correlated"])
    assert any("constant column 'constant'" in problem for problem in problems)
    assert any("both" in problem for problem in problems)
    assert any("'correlated', which the screen does not flag" in problem for problem in problems)
    assert any("neither" in problem for problem in check_screen(screen, ["constant"]))
