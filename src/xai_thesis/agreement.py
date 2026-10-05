"""Feature agreement and rank agreement between SHAP and LIME (local explanations only).

Implements the two top-k disagreement metrics of:

    Krishna et al., "The Disagreement Problem in Explainable Machine Learning:
    A Practitioner's Perspective", TMLR 2024 (arXiv:2202.01602).

Definitions (per instance, in [0, 1], higher = more agreement), with T_k(A) the set of the k
features with the largest |attribution| under method A:

  feature agreement  FA_k = |T_k(A) n T_k(B)| / k

  rank agreement     RA_k = |{f in T_k(A) n T_k(B) : rank_A(f) == rank_B(f)}| / k

rank_A(f) is f's 1-based position when every feature is ordered by |attribution| descending.
Ties in |attribution| are broken deterministically by feature name (ascending), or by column
index when no names are given (see ranked_feature_indices), so every ordering, and therefore
every T_k and every rank, is unique and reproducible. A shared feature counts toward RA_k only if
it sits at the same position in both orderings, so RA_k <= FA_k always, and agreement_fraction
asserts it.

Unlike the authors' reference implementation (AI4LIFE-GROUP/disagreement-problem), which leaves
ties to np.argsort's unstable order, ties are broken deterministically. For the attributions
analysed here no two features tie within the top 10, so the results are the same.

Both functions take two signed attribution matrices of shape [n_points, n_features] for the
*same* points, in the same feature order, and return one value per point.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

AGREEMENT_METRIC_TITLES = {"feature": "Feature agreement", "rank": "Rank agreement"}
AGREEMENT_METRICS = tuple(AGREEMENT_METRIC_TITLES)


def ranked_feature_indices(attributions: np.ndarray, feature_names: Sequence[str] | None = None) -> np.ndarray:
    """Feature indices per row, from largest to smallest |attribution|.

    Ties in |attribution| go to the feature whose name sorts first (plain string order), or to
    the lower column index when feature_names is None. Accepts [n_features] or
    [n_points, n_features] and returns the same shape.
    """
    magnitudes = np.abs(np.asarray(attributions, dtype=float))
    n_features = magnitudes.shape[-1]
    if feature_names is None:
        tiebreak = np.arange(n_features)
    else:
        if len(feature_names) != n_features:
            raise ValueError(f"{len(feature_names)} feature names for {n_features} features")
        tiebreak = np.empty(n_features, dtype=int)
        tiebreak[np.argsort(np.asarray(feature_names, dtype=str), kind="stable")] = np.arange(n_features)
    # np.lexsort sorts by the last key first: primary -|attribution|, secondary the name order.
    return np.lexsort((np.broadcast_to(tiebreak, magnitudes.shape), -magnitudes), axis=-1)


def agreement_fraction(
    attr_a: np.ndarray,
    attr_b: np.ndarray,
    k: int,
    metric: str,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Per-point top-k agreement between two [n_points, n_features] attribution matrices.

    metric is "feature" or "rank" (see module docstring). feature_names sets the tie-break.
    Returns shape [n_points].
    """
    if metric not in AGREEMENT_METRIC_TITLES:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(AGREEMENT_METRIC_TITLES)}")
    attr_a = np.atleast_2d(np.asarray(attr_a, dtype=float))
    attr_b = np.atleast_2d(np.asarray(attr_b, dtype=float))
    if attr_a.shape != attr_b.shape:
        raise ValueError(f"shape mismatch: {attr_a.shape} vs {attr_b.shape}")
    k = min(k, attr_a.shape[1])

    topk_a = ranked_feature_indices(attr_a, feature_names)[:, :k]
    topk_b = ranked_feature_indices(attr_b, feature_names)[:, :k]

    feature = np.array([len(set(row_a) & set(row_b)) / k for row_a, row_b in zip(topk_a, topk_b)])
    # Same feature at the same position in both orderings: shared, and with equal rank.
    rank = (topk_a == topk_b).sum(axis=1) / k
    assert (rank <= feature).all(), "rank agreement exceeded feature agreement"
    return feature if metric == "feature" else rank


def feature_agreement(
    attr_a: np.ndarray, attr_b: np.ndarray, k: int, feature_names: Sequence[str] | None = None
) -> np.ndarray:
    """Per-point feature agreement, shape [n_points]."""
    return agreement_fraction(attr_a, attr_b, k, "feature", feature_names)


def rank_agreement(
    attr_a: np.ndarray, attr_b: np.ndarray, k: int, feature_names: Sequence[str] | None = None
) -> np.ndarray:
    """Per-point rank agreement, shape [n_points]."""
    return agreement_fraction(attr_a, attr_b, k, "rank", feature_names)


def mean_and_standard_error(values: np.ndarray) -> tuple[float, float]:
    """Mean and standard error, SE = sample standard deviation (ddof=1) / sqrt(n)."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float(values.mean()), float("nan")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def agreement_over_k(
    attr_a: np.ndarray,
    attr_b: np.ndarray,
    k_values: Sequence[int],
    metric: str,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of the metric across points, one entry per k in k_values."""
    stats = [
        mean_and_standard_error(agreement_fraction(attr_a, attr_b, k, metric, feature_names)) for k in k_values
    ]
    return np.array([mean for mean, _ in stats]), np.array([sem for _, sem in stats])


def summarize_agreement(
    attr_a: np.ndarray,
    attr_b: np.ndarray,
    k_values: Sequence[int],
    feature_names: Sequence[str] | None = None,
) -> list[dict[str, float]]:
    """One row per k: FA and RA mean and standard error across points, plus n."""
    rows = []
    for k in k_values:
        fa_mean, fa_se = mean_and_standard_error(agreement_fraction(attr_a, attr_b, k, "feature", feature_names))
        ra_mean, ra_se = mean_and_standard_error(agreement_fraction(attr_a, attr_b, k, "rank", feature_names))
        rows.append(
            {"k": k, "fa_mean": fa_mean, "fa_se": fa_se, "ra_mean": ra_mean, "ra_se": ra_se, "n": len(attr_a)}
        )
    return rows


def plot_agreement_heatmap(
    values: np.ndarray,
    sems: np.ndarray,
    row_labels: Sequence[str],
    k_values: Sequence[int],
    output_path: str | Path,
    *,
    metric: str,
    title: str,
    footnote: str = "",
) -> Path:
    """Annotated heatmap of SHAP-vs-LIME agreement, rows = models, columns = k.

    Fixed [0, 1] colour scale so that panels are comparable, the value annotated in each cell,
    and the minimum and maximum standard error printed underneath. The sequential colormap is
    light for low and dark for high agreement. footnote adds a line under the plot, e.g. to name
    models that were left out.
    """
    fig, ax = plt.subplots(figsize=(1.5 * len(k_values) + 3.5, 0.75 * len(row_labels) + 2.5))
    sns.heatmap(
        values,
        cmap=sns.color_palette("Reds", as_cmap=True),
        vmin=0,
        vmax=1,
        xticklabels=[f"k = {k}" for k in k_values],
        yticklabels=list(row_labels),
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": AGREEMENT_METRIC_TITLES[metric]},
        ax=ax,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title)

    caption = (
        f"SHAP vs LIME, mean over instances. "
        f"Standard errors: min={np.nanmin(sems):.3f}, max={np.nanmax(sems):.3f}"
    )
    if footnote:
        caption = f"{caption}\n{footnote}"
    fig.text(0.5, 0.01, caption, ha="center", fontsize="small")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
