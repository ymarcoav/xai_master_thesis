"""Feature agreement and rank agreement between two explanation methods.

Implements the two top-k disagreement metrics of:

    Krishna et al., "The Disagreement Problem in Explainable Machine Learning:
    A Practitioner's Perspective", TMLR 2024 (arXiv:2202.01602).

`agreement_fraction` is a direct port of `agreement_fraction()` from the authors'
reference implementation (AI4LIFE-GROUP/disagreement-problem,
experiments/tabular/metrics.py) - vectorised over [n_points, n_features], with the
original's removed-in-pandas-3.0 `DataFrame.applymap(str)` written as
`DataFrame.astype(str)` (identical element-wise). The paper defines two further
metrics, sign agreement and signed rank agreement, which are not implemented here: the
heatmaps use only these two. Sign agreement is computed per instance by
explanation_report.sign_agreement; signed rank agreement is not used.

Definitions (per data point, in [0, 1], higher = more agreement):

  feature agreement - fraction of the top-k features (ranked by |attribution|)
      the two explanations have in common: |topk(A) n topk(B)| / k. This is the
      metric 84% of the paper's interviewed practitioners named as the single most
      important thing they read off an explanation.

  rank agreement - fraction of the top-k features that are both shared *and* given
      the same rank by both explanations. Ranks are computed over all features with
      scipy's 'dense' method so ties are handled. Stricter than feature agreement,
      and <= it by construction.

Both take two signed attribution matrices of shape [n_points, n_features] for the
*same* points, and return one value per point; average for a scalar.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from scipy.stats import sem as standard_error  # noqa: E402

AGREEMENT_METRIC_TITLES = {"feature": "Feature agreement", "rank": "Rank agreement"}
AGREEMENT_METRICS = tuple(AGREEMENT_METRIC_TITLES)


def agreement_fraction(attr_a: np.ndarray, attr_b: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Per-point top-k agreement between two [n_points, n_features] attribution matrices.

    metric is "feature" or "rank" (see module docstring). Returns shape [n_points].
    """
    if metric not in AGREEMENT_METRIC_TITLES:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(AGREEMENT_METRIC_TITLES)}")
    attr_a = np.asarray(attr_a, dtype=float)
    attr_b = np.asarray(attr_b, dtype=float)
    if attr_a.shape != attr_b.shape:
        raise ValueError(f"shape mismatch: {attr_a.shape} vs {attr_b.shape}")
    k = min(k, attr_a.shape[1])

    # Top-k feature ids, most important (largest |attribution|) first.
    topk_id_a = np.argsort(-np.abs(attr_a), axis=1)[:, :k]
    topk_id_b = np.argsort(-np.abs(attr_b), axis=1)[:, :k]

    if metric == "feature":
        return np.array(
            [len(set(row_a) & set(row_b)) / k for row_a, row_b in zip(topk_id_a, topk_id_b)]
        )

    # "rank": a feature counts only if it sits at the same position in both top-k lists
    # AND carries the same dense rank over all features (rankdata ranks ascending, so
    # rank -|attr| to give rank 1 to the largest magnitude and let ties share a rank).
    ranks_a = np.take_along_axis(rankdata(-np.abs(attr_a), method="dense", axis=1), topk_id_a, axis=1)
    ranks_b = np.take_along_axis(rankdata(-np.abs(attr_b), method="dense", axis=1), topk_id_b, axis=1)
    id_rank_a = ("feat" + pd.DataFrame(topk_id_a).astype(str)) + ("rank" + pd.DataFrame(ranks_a).astype(str))
    id_rank_b = ("feat" + pd.DataFrame(topk_id_b).astype(str)) + ("rank" + pd.DataFrame(ranks_b).astype(str))
    return (id_rank_a == id_rank_b).sum(axis=1).to_numpy() / k


def feature_agreement(attr_a: np.ndarray, attr_b: np.ndarray, k: int) -> np.ndarray:
    """Per-point feature agreement, shape [n_points]."""
    return agreement_fraction(attr_a, attr_b, k, metric="feature")


def rank_agreement(attr_a: np.ndarray, attr_b: np.ndarray, k: int) -> np.ndarray:
    """Per-point rank agreement, shape [n_points]."""
    return agreement_fraction(attr_a, attr_b, k, metric="rank")


def agreement_over_k(
    attr_a: np.ndarray, attr_b: np.ndarray, k_values: Sequence[int], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of the metric across points, one entry per k in k_values."""
    distributions = [agreement_fraction(attr_a, attr_b, k, metric) for k in k_values]
    means = np.array([distribution.mean() for distribution in distributions])
    sems = np.array([standard_error(distribution) for distribution in distributions])
    return means, sems


def plot_agreement_heatmap(
    values: np.ndarray,
    sems: np.ndarray | None,
    row_labels: Sequence[str],
    k_values: Sequence[int],
    output_path: str | Path,
    *,
    metric: str,
    title: str,
    pair_label: str = "SHAP vs LIME",
    footnote: str = "",
) -> Path:
    """Annotated heatmap of agreement values, rows = models, columns = k.

    Follows the reference implementation's `heatmap_metric_avg`: seaborn heatmap on a
    fixed [0, 1] colour scale (so panels are comparable), value annotated in each cell,
    and the min/max standard error printed underneath. Deviations from the reference:
    the axes are models x k rather than method x method (this project has only two
    per-instance methods, SHAP and LIME, so the method-pair matrix would be 2x2), and
    dpi is this project's usual 150 rather than the reference's 2000.

    The colormap is sequential "Reds", light for low agreement and dark for high, matching
    the paper's own convention (its Figure 1 caption: "lower values (lighter colors)
    indicating stronger disagreement"). A diverging map would be wrong here regardless of
    taste: agreement runs 0 to 1 with no meaningful midpoint, so the white band a diverging
    palette puts at 0.5 would imply a neutral point that does not exist.

    sems is None for the global (SHAP vs permutation importance) panels: those compare two
    single dataset-level rankings, so there is one value per cell and no across-instance
    spread to report. footnote adds a further line under the plot - used to record which
    model/dataset cells were left out and why.
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

    if sems is None:
        caption = f"{pair_label}, one ranking pair per model (no across-instance spread)."
    else:
        caption = (
            f"{pair_label}, mean over instances. "
            f"Standard errors: min={sems.min():.3f}, max={sems.max():.3f}"
        )
    if footnote:
        caption = f"{caption}\n{footnote}"
    fig.text(0.5, 0.01, caption, ha="center", fontsize="small")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
