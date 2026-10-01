import numpy as np
import pytest

from xai_thesis.agreement import (
    AGREEMENT_METRICS,
    agreement_fraction,
    agreement_over_k,
    feature_agreement,
    mean_and_standard_error,
    plot_agreement_heatmap,
    rank_agreement,
    ranked_feature_indices,
    summarize_agreement,
)


def test_identical_explanations_agree_perfectly() -> None:
    attributions = np.array([[3.0, -2.0, 1.0, 0.5], [0.1, 4.0, -3.0, 2.0]])
    assert feature_agreement(attributions, attributions, k=2).tolist() == [1.0, 1.0]
    assert rank_agreement(attributions, attributions, k=2).tolist() == [1.0, 1.0]


def test_feature_agreement_ignores_sign_and_counts_shared_top_k() -> None:
    # top-2 by |value|: A -> {0, 1}, B -> {0, 2}; one shared feature out of two.
    attr_a = np.array([[10.0, 9.0, 1.0, 0.5]])
    attr_b = np.array([[-10.0, 0.0, -9.0, 0.5]])
    assert feature_agreement(attr_a, attr_b, k=2).tolist() == [0.5]


def test_feature_agreement_is_zero_for_disjoint_top_k() -> None:
    attr_a = np.array([[5.0, 4.0, 1.0, 0.0]])
    attr_b = np.array([[0.0, 1.0, 4.0, 5.0]])
    assert feature_agreement(attr_a, attr_b, k=2).tolist() == [0.0]


def test_rank_agreement_is_stricter_than_feature_agreement() -> None:
    # Same top-2 set {0, 1} but swapped order, so features agree and ranks do not.
    attr_a = np.array([[9.0, 8.0, 1.0]])
    attr_b = np.array([[8.0, 9.0, 1.0]])
    assert feature_agreement(attr_a, attr_b, k=2).tolist() == [1.0]
    assert rank_agreement(attr_a, attr_b, k=2).tolist() == [0.0]


def test_rank_agreement_counts_shared_features_at_the_same_rank() -> None:
    # Top-3: A = [0, 1, 2], B = [0, 2, 1] -> only feature 0 holds the same rank.
    attr_a = np.array([[9.0, 8.0, 7.0, 1.0]])
    attr_b = np.array([[9.0, 7.0, 8.0, 1.0]])
    assert feature_agreement(attr_a, attr_b, k=3).tolist() == [1.0]
    assert rank_agreement(attr_a, attr_b, k=3).tolist() == pytest.approx([1 / 3])


def test_rank_agreement_never_exceeds_feature_agreement() -> None:
    rng = np.random.default_rng(0)
    attr_a = rng.normal(size=(50, 12))
    attr_b = attr_a + rng.normal(scale=0.5, size=attr_a.shape)
    for k in (1, 3, 5, 10):
        assert (rank_agreement(attr_a, attr_b, k) <= feature_agreement(attr_a, attr_b, k)).all()


def test_ties_are_broken_by_feature_name_not_column_order() -> None:
    names = ["zeta", "alpha", "mid"]
    # |zeta| == |alpha|: alpha sorts first by name, so it takes rank 1.
    order = ranked_feature_indices(np.array([[-2.0, 2.0, 1.0]]), names)
    assert order.tolist() == [[1, 0, 2]]
    # Without names the lower column index wins.
    assert ranked_feature_indices(np.array([[-2.0, 2.0, 1.0]])).tolist() == [[0, 1, 2]]


def test_tie_break_makes_top_k_deterministic_for_both_methods() -> None:
    names = ["b", "a", "c"]
    attr_a = np.array([[1.0, 1.0, 0.0]])  # tie between b and a -> a first
    attr_b = np.array([[0.5, 1.0, 0.0]])  # a first outright
    assert rank_agreement(attr_a, attr_b, k=1, feature_names=names).tolist() == [1.0]
    # By column index instead, attr_a's top-1 would be b, and the ranks would disagree.
    assert rank_agreement(attr_a, attr_b, k=1).tolist() == [0.0]


def test_k_is_capped_at_the_feature_count() -> None:
    attributions = np.array([[2.0, 1.0]])
    assert feature_agreement(attributions, attributions, k=99).tolist() == [1.0]


def test_agreement_fraction_rejects_unknown_metric_and_shape_mismatch() -> None:
    attributions = np.zeros((2, 3))
    with pytest.raises(ValueError):
        agreement_fraction(attributions, attributions, k=2, metric="sign")
    with pytest.raises(ValueError):
        agreement_fraction(attributions, np.zeros((2, 4)), k=2, metric="feature")


def test_standard_error_uses_the_sample_standard_deviation() -> None:
    mean, se = mean_and_standard_error(np.array([0.0, 1.0, 1.0, 0.0]))
    assert mean == 0.5
    assert se == pytest.approx(np.std([0, 1, 1, 0], ddof=1) / 2)


@pytest.mark.parametrize("metric", AGREEMENT_METRICS)
def test_agreement_over_k_returns_one_mean_and_sem_per_k(metric: str) -> None:
    rng = np.random.default_rng(1)
    attr_a = rng.normal(size=(30, 8))
    attr_b = rng.normal(size=(30, 8))
    k_values = [1, 3, 5]

    means, sems = agreement_over_k(attr_a, attr_b, k_values, metric)

    assert means.shape == sems.shape == (len(k_values),)
    assert ((means >= 0) & (means <= 1)).all()


def test_summarize_agreement_reports_fa_ra_mean_se_and_n_per_k() -> None:
    rng = np.random.default_rng(2)
    attr_a = rng.normal(size=(20, 6))
    attr_b = rng.normal(size=(20, 6))

    rows = summarize_agreement(attr_a, attr_b, [1, 5])

    assert [row["k"] for row in rows] == [1, 5]
    for row in rows:
        assert set(row) == {"k", "fa_mean", "fa_se", "ra_mean", "ra_se", "n"}
        assert row["n"] == 20
        assert row["ra_mean"] <= row["fa_mean"]


def test_plot_agreement_heatmap_writes_png(tmp_path) -> None:
    values = np.array([[0.2, 0.4], [0.5, 0.6]])
    sems = np.full_like(values, 0.05)

    output_path = plot_agreement_heatmap(
        values,
        sems,
        ["random_forest", "xgboost"],
        [3, 5],
        tmp_path / "local_agreement_feature.png",
        metric="feature",
        title="Feature agreement - SHAP vs LIME, test",
    )

    assert output_path.exists()
