import numpy as np
import pandas as pd
import pytest

from xai_thesis.explanation_report import (
    build_local_report,
    global_overlap_table,
    global_top_table,
    local_agreement_table,
    rank_positions,
    sign_agreement,
    top_n_disagreements,
    top_n_rows,
)

NAMES = ["a", "b", "c", "d", "e"]


def test_top_n_rows_ranks_by_magnitude_and_keeps_the_sign() -> None:
    rows = top_n_rows(np.array([0.1, -0.9, 0.5, 0.0, 0.2]), NAMES, n=3)

    assert [(row["rank"], row["feature"], row["value"]) for row in rows] == [
        (1, "b", -0.9),
        (2, "c", 0.5),
        (3, "e", 0.2),
    ]


def test_rank_positions_are_one_based_and_break_ties_by_feature_name() -> None:
    assert rank_positions(np.array([0.1, -0.9, 0.5, 0.0, 0.2]), NAMES).tolist() == [4, 1, 2, 5, 3]
    # |values| tie between "e" and "b": "b" sorts first, whatever the column order.
    assert rank_positions(np.array([0.9, 0.0, 0.1, 0.0, -0.5]), ["z", "e", "y", "b", "x"]).tolist() == [1, 5, 3, 4, 2]


def test_sign_agreement_counts_only_shared_top_k_features_with_the_same_sign() -> None:
    shap = np.array([0.9, 0.8, -0.7, 0.0, 0.0])
    lime = np.array([0.9, -0.8, -0.7, 0.0, 0.0])

    # Top-3 sets are identical; b flips sign, so 2 of 3 agree.
    assert sign_agreement(shap, lime, 3, NAMES) == pytest.approx(2 / 3)


def test_top_n_disagreements_reports_rank_in_the_other_method_and_sign_flips() -> None:
    shap = np.array([0.9, 0.8, -0.7, 0.05, 0.0])
    lime = np.array([0.9, -0.8, 0.01, 0.6, 0.0])

    rows = top_n_disagreements(shap, lime, NAMES, n=3)
    by_kind = {(row["kind"], row["feature"]): row for row in rows}

    # c is SHAP's #3 but only LIME's #4; d is LIME's #3 but SHAP's #4.
    assert by_kind[("only_in_shap_top3", "c")]["lime_rank"] == 4
    assert by_kind[("only_in_lime_top3", "d")]["shap_rank"] == 4
    # b is in both top-3s with opposite signs.
    assert by_kind[("sign_disagreement", "b")]["shap_value"] == 0.8
    assert by_kind[("sign_disagreement", "b")]["lime_value"] == -0.8
    assert len(rows) == 3


def test_top_n_disagreements_blanks_the_rank_of_a_feature_the_other_method_ignores() -> None:
    shap = np.array([0.9, 0.8, 0.7, 0.0, 0.0])
    lime = np.array([0.9, 0.8, 0.0, 0.5, 0.0])

    rows = top_n_disagreements(shap, lime, NAMES, n=3)
    only_shap = next(row for row in rows if row["kind"] == "only_in_shap_top3")

    # c has zero LIME attribution: its position among the tied zeros means nothing.
    assert only_shap["feature"] == "c"
    assert only_shap["lime_rank"] is None


def _write_global_cache(path, importance) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        importance=np.array(importance, dtype=float),
        feature_names=np.array(NAMES),
        instance_index=np.arange(10),
        class_names=np.array(["Benign", "Attack"]),
        class_counts=np.array([5, 5]),
    )


def test_global_top_table_lists_each_method_descriptively_and_skips_missing_caches(tmp_path) -> None:
    cache_dir = tmp_path / "metrics" / "global_importance"
    _write_global_cache(cache_dir / "toy_m1_baseline__shap.npz", [5, 4, 3, 2, 1])
    # A negative PFI ranks below zero (no measurable importance), not by its magnitude.
    _write_global_cache(cache_dir / "toy_m1_baseline__pfi.npz", [-9, 4, 3, 0, 1])
    # m2's global SHAP was skipped: only its PFI is cached.
    _write_global_cache(cache_dir / "toy_m2_baseline__pfi.npz", [1, 1, 3, 0, 0])

    top = global_top_table(["toy"], ["m1", "m2"], metrics_dir=tmp_path / "metrics", n=3)

    assert list(top.columns) == ["dataset", "model", "method", "rank", "feature", "importance", "n_instances"]
    m1_pfi = top[(top["model"] == "m1") & (top["method"] == "PFI")]
    assert m1_pfi["feature"].tolist() == ["b", "c", "e"]
    assert m1_pfi["rank"].tolist() == [1, 2, 3]
    # Tie between a and b broken by name.
    assert top[top["model"] == "m2"]["feature"].tolist() == ["c", "a", "b"]
    assert set(top[top["model"] == "m2"]["method"]) == {"PFI"}
    assert set(top["n_instances"]) == {10}


def test_global_overlap_table_counts_shared_top_k_features_and_skips_missing_caches(tmp_path) -> None:
    cache_dir = tmp_path / "metrics" / "global_importance"
    _write_global_cache(cache_dir / "toy_m1_baseline__shap.npz", [5, 4, 3, 2, 1])
    # a's negative PFI keeps it out of the PFI top 3 despite its magnitude.
    _write_global_cache(cache_dir / "toy_m1_baseline__pfi.npz", [-9, 4, 3, 0, 1])
    _write_global_cache(cache_dir / "toy_m2_baseline__pfi.npz", [1, 1, 3, 0, 0])

    overlap = global_overlap_table(["toy"], ["m1", "m2"], metrics_dir=tmp_path / "metrics", k=3)

    assert list(overlap.columns) == ["dataset", "model", "k", "n_shared", "feature_agreement", "shared"]
    # SHAP top 3 = a, b, c; PFI top 3 = b, c, e.
    assert overlap.to_dict("records") == [
        {"dataset": "toy", "model": "m1", "k": 3, "n_shared": 2, "feature_agreement": pytest.approx(2 / 3),
         "shared": "b; c"}
    ]


def test_local_agreement_table_has_the_thesis_columns_and_requires_one_instance_set(tmp_path) -> None:
    attributions = tmp_path / "metrics" / "agreement" / "attributions"
    attributions.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for model, index in (("m1", [1, 2, 3]), ("m2", [1, 2, 3]), ("m3", [1, 2, 4])):
        np.savez(
            attributions / f"toy_{model}_baseline.npz",
            shap=rng.normal(size=(3, 5)),
            lime=rng.normal(size=(3, 5)),
            feature_names=np.array(NAMES),
            instance_index=np.array(index),
        )

    table = local_agreement_table(["toy"], ["m1", "m2"], metrics_dir=tmp_path / "metrics", k_values=(1, 3))

    assert list(table.columns) == ["dataset", "model", "k", "fa_mean", "fa_se", "ra_mean", "ra_se", "n"]
    assert table["k"].tolist() == [1, 3, 1, 3]
    assert (table["n"] == 3).all()
    assert (table["ra_mean"] <= table["fa_mean"]).all()
    with pytest.raises(ValueError, match="instance set differs"):
        local_agreement_table(["toy"], ["m1", "m3"], metrics_dir=tmp_path / "metrics")


def test_build_local_report_writes_per_instance_overlap_and_checks_cache_consistency(tmp_path) -> None:
    metrics_dir = tmp_path / "metrics"
    attributions = metrics_dir / "agreement" / "attributions"
    attributions.mkdir(parents=True)
    np.savez(
        attributions / "toy_m1_baseline.npz",
        shap=np.array([[0.9, 0.8, -0.7, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4, 0.5]]),
        lime=np.array([[0.9, -0.8, -0.7, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4, 0.5]]),
        feature_names=np.array(NAMES),
        instance_index=np.array([10, 20]),
        explained_class=np.array(["Attack", "Benign"]),
    )
    (metrics_dir / "stimuli").mkdir()
    pd.DataFrame(
        {
            "model": ["m1", "m1"],
            "instance_index": [10, 20],
            "true_class": ["Attack", "Attack"],
            "predicted_class": ["Attack", "Benign"],
            "error_type": ["correct", "false_negative"],
            "proba_argmax_class": ["Attack", "Benign"],
            "predict_disagrees_with_proba": [False, False],
        }
    ).to_csv(metrics_dir / "stimuli" / "toy_error_profile.csv", index=False)

    paths = build_local_report("toy", ["m1"], metrics_dir=metrics_dir, output_dir=tmp_path / "out", n=3, k_values=(3,))
    instances = pd.read_csv(paths["local_instances"]).set_index("instance_index")
    summary = pd.read_csv(paths["local_summary"]).set_index("subset")

    assert instances.loc[10, "sign_disagreements_top3"] == 1
    assert instances.loc[10, "sign_agreement@3"] == pytest.approx(2 / 3)
    assert instances.loc[20, "rank_agreement@3"] == 1.0
    assert not instances.loc[20, "correct"]
    assert summary.loc["misclassified", "n_instances"] == 1


def test_build_local_report_rejects_an_explained_class_that_disagrees_with_the_profile(tmp_path) -> None:
    metrics_dir = tmp_path / "metrics"
    attributions = metrics_dir / "agreement" / "attributions"
    attributions.mkdir(parents=True)
    np.savez(
        attributions / "toy_m1_baseline.npz",
        shap=np.ones((1, 5)),
        lime=np.ones((1, 5)),
        feature_names=np.array(NAMES),
        instance_index=np.array([10]),
        explained_class=np.array(["Benign"]),
    )
    (metrics_dir / "stimuli").mkdir()
    pd.DataFrame(
        {
            "model": ["m1"],
            "instance_index": [10],
            "true_class": ["Attack"],
            "predicted_class": ["Attack"],
            "error_type": ["correct"],
            "proba_argmax_class": ["Attack"],
            "predict_disagrees_with_proba": [False],
        }
    ).to_csv(metrics_dir / "stimuli" / "toy_error_profile.csv", index=False)

    with pytest.raises(ValueError, match="out of sync"):
        build_local_report("toy", ["m1"], metrics_dir=metrics_dir, output_dir=tmp_path / "out", n=3)
