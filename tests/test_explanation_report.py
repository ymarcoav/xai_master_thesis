import numpy as np
import pandas as pd
import pytest

from xai_thesis.explanation_report import (
    build_global_report,
    build_local_report,
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


def test_rank_positions_are_one_based_positions_in_the_magnitude_order() -> None:
    assert rank_positions(np.array([0.1, -0.9, 0.5, 0.0, 0.2])).tolist() == [4, 1, 2, 5, 3]


def test_sign_agreement_counts_only_shared_top_k_features_with_the_same_sign() -> None:
    shap = np.array([0.9, 0.8, -0.7, 0.0, 0.0])
    lime = np.array([0.9, -0.8, -0.7, 0.0, 0.0])

    # Top-3 sets are identical; b flips sign, so 2 of 3 agree.
    assert sign_agreement(shap, lime, 3) == pytest.approx(2 / 3)


def test_top_n_disagreements_reports_rank_in_the_other_method_and_sign_flips() -> None:
    shap = np.array([0.9, 0.8, -0.7, 0.05, 0.0])
    lime = np.array([0.9, -0.8, 0.01, 0.6, 0.0])

    rows = top_n_disagreements(shap, lime, NAMES, "shap", "lime", n=3, signed=True)
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
    permutation = np.array([0.9, 0.8, 0.0, 0.5, 0.0])

    rows = top_n_disagreements(shap, permutation, NAMES, "shap", "permutation", n=3, signed=False)
    only_shap = next(row for row in rows if row["kind"] == "only_in_shap_top3")

    # c has zero permutation importance: its position among the tied zeros means nothing.
    assert only_shap["feature"] == "c"
    assert only_shap["permutation_rank"] is None


def _write_global_cache(path, shap, permutation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        shap=np.array(shap),
        permutation=np.array(permutation),
        feature_names=np.array(NAMES),
        class_names=np.array(["Benign", "Attack"]),
        class_counts=np.array([5, 5]),
    )


def test_build_global_report_compares_methods_and_every_model_pair_and_notes_missing_models(tmp_path) -> None:
    cache_dir = tmp_path / "metrics" / "agreement" / "global"
    _write_global_cache(cache_dir / "toy_m1_baseline__balanced500.npz", [5, 4, 3, 2, 1], [5, 4, 3, 2, -1])
    _write_global_cache(cache_dir / "toy_m2_baseline__balanced500.npz", [1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    _write_global_cache(cache_dir / "toy_m3_baseline__balanced500.npz", [5, 4, 3, 2, 1], [1, 2, 3, 4, 5])

    paths = build_global_report(
        "toy", ["m1", "m2", "m3", "m4"], metrics_dir=tmp_path / "metrics", output_dir=tmp_path / "out", n=3,
        k_values=(3,),
    )
    overlap = pd.read_csv(paths["global_overlap"])
    top = pd.read_csv(paths["global_top"])

    methods = overlap[overlap["comparison"] == "methods"].set_index("model_a")
    assert methods.loc["m1", "feature_agreement"] == 1.0
    assert methods.loc["m2", "feature_agreement"] == pytest.approx(1 / 3)
    models = overlap[overlap["comparison"] == "models"]
    assert len(models) == 2 * 3  # two methods x all three pairs of the cached models
    assert set(overlap["missing_models"]) == {"m4"}
    # The raw (negative) permutation value is reported, even though ranking uses the clipped one.
    assert top[(top["model"] == "m1") & (top["method"] == "permutation")]["feature"].tolist() == ["a", "b", "c"]


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
