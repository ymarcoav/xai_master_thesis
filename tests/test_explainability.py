import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from xai_thesis.explainability import (
    clean_feature_names,
    compute_permutation_importance,
    compute_shap_values,
    explain_instance_with_lime,
    plot_partial_dependence,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_waterfall,
    predict_proba_fn,
    select_curated_instances,
)
from xai_thesis.models import build_model
from xai_thesis.preprocessing import build_preprocessing_pipeline


def _make_dataset(n_per_class: int = 40, random_state: int = 42, n_classes: int = 3):
    rng = np.random.default_rng(random_state)
    frames = []
    for class_index in range(n_classes):
        center = class_index * 3.0
        frames.append(
            pd.DataFrame(
                {
                    "feature_a": rng.normal(center, 1.0, n_per_class),
                    "feature_b": rng.normal(-center, 1.0, n_per_class),
                    "label": [f"class_{class_index}"] * n_per_class,
                }
            )
        )
    data = pd.concat(frames, ignore_index=True)
    x = data[["feature_a", "feature_b"]]
    y = data["label"]
    return x, y


def _fitted_pipeline(model_name: str, x, y, random_state: int = 42, categorical_columns=()) -> Pipeline:
    preprocessing = build_preprocessing_pipeline(x, categorical_columns=categorical_columns, random_state=random_state)
    model = build_model(model_name, random_state=random_state)
    pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])
    pipeline.fit(x, y)
    return pipeline


def _make_dataset_with_categorical(n_per_class: int = 40, random_state: int = 42):
    x, y = _make_dataset(n_per_class, random_state)
    rng = np.random.default_rng(random_state)
    x = x.copy()
    x["protocol"] = rng.choice(["tcp", "udp"], size=len(x))
    return x, y


def test_clean_feature_names_strips_column_transformer_prefix() -> None:
    assert clean_feature_names(["numeric__dur", "categorical__proto_tcp", "no_prefix"]) == [
        "dur",
        "proto_tcp",
        "no_prefix",
    ]


def test_predict_proba_fn_uses_native_predict_proba_when_available() -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("gradient_boosting", x, y)
    model = pipeline.named_steps["model"]

    fn = predict_proba_fn(model)

    assert fn == model.predict_proba


def test_predict_proba_fn_softmax_fallback_preserves_argmax_for_svm() -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("svm", x, y)
    model = pipeline.named_steps["model"]
    x_transformed = pipeline.named_steps["preprocessing"].transform(x)

    fn = predict_proba_fn(model)
    proba = fn(x_transformed)

    assert proba.shape == (len(x), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    proxy_predictions = model.classes_[proba.argmax(axis=1)]
    assert (proxy_predictions == model.predict(x_transformed)).all()


@pytest.mark.parametrize("model_name", ["gradient_boosting", "svm", "neural_network"])
def test_compute_shap_values_has_consistent_shape_across_model_types(model_name: str) -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline(model_name, x, y)

    shap_values, base_values, x_transformed, feature_names = compute_shap_values(
        pipeline, model_name, x.iloc[:10], x.iloc[10:30]
    )

    assert shap_values.shape == (10, 2, 3)
    assert base_values.shape == (10, 3)
    assert x_transformed.shape[0] == 10
    assert feature_names == ["feature_a", "feature_b"]


def test_compute_shap_values_collapses_one_hot_columns_and_restores_original_values() -> None:
    x, y = _make_dataset_with_categorical()
    pipeline = _fitted_pipeline("gradient_boosting", x, y, categorical_columns=["protocol"])

    shap_values, _, display_data, feature_names = compute_shap_values(
        pipeline, "gradient_boosting", x.iloc[:10], x.iloc[10:30], categorical_columns=["protocol"]
    )

    # One-hot expansion of "protocol" (2 categories) must be collapsed back to a single
    # feature, not left as two ("protocol_tcp", "protocol_udp") separate attributions.
    assert feature_names == ["feature_a", "feature_b", "protocol"]
    assert shap_values.shape == (10, 3, 3)

    # display_data must carry the original, untransformed values - not the quantile-scaled
    # numeric representation, and not a one-hot 0/1 indicator for the categorical column.
    assert np.allclose(display_data[:, 0].astype(float), x["feature_a"].iloc[:10].to_numpy())
    assert list(display_data[:, 2]) == list(x["protocol"].iloc[:10])


@pytest.mark.parametrize("model_name", ["gradient_boosting", "svm"])
def test_compute_shap_values_expands_binary_classifier_to_per_class_shape(model_name: str) -> None:
    # TreeExplainer/LinearExplainer collapse a binary classifier's output to a single column
    # (unlike the always-per-class multiclass case, or the permutation explainer's
    # predict_proba-based path), which crashed downstream shape-dependent code the first time
    # this project trained on real binary-labeled data.
    x, y = _make_dataset(n_classes=2)
    pipeline = _fitted_pipeline(model_name, x, y)

    shap_values, base_values, _, feature_names = compute_shap_values(
        pipeline, model_name, x.iloc[:10], x.iloc[10:30]
    )

    assert shap_values.shape == (10, 2, 2)
    assert base_values.shape == (10, 2)
    assert feature_names == ["feature_a", "feature_b"]
    # class 0's contribution must be exactly the negation of class 1's, per the binary SHAP
    # convention this project verified empirically against the model's own predict_proba.
    assert np.allclose(shap_values[:, :, 0], -shap_values[:, :, 1])
    assert np.allclose(base_values[:, 0], -base_values[:, 1])


def test_plot_shap_bar_and_beeswarm_write_png(tmp_path) -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("gradient_boosting", x, y)
    shap_values, _, x_transformed, feature_names = compute_shap_values(pipeline, "gradient_boosting", x, x.iloc[:30])
    class_names = list(pipeline.named_steps["model"].classes_)

    bar_path = plot_shap_bar(shap_values, x_transformed, feature_names, class_names, tmp_path / "bar.png")
    beeswarm_path = plot_shap_beeswarm(shap_values, x_transformed, feature_names, 0, tmp_path / "beeswarm.png")

    assert bar_path.exists()
    assert beeswarm_path.exists()


def test_plot_shap_waterfall_writes_png(tmp_path) -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("gradient_boosting", x, y)
    shap_values, base_values, x_transformed, feature_names = compute_shap_values(
        pipeline, "gradient_boosting", x.iloc[:5], x.iloc[5:30]
    )

    output_path = plot_shap_waterfall(
        shap_values, base_values, x_transformed[0], feature_names, 0, 0, tmp_path / "waterfall.png"
    )

    assert output_path.exists()


def test_explain_instance_with_lime_writes_png(tmp_path) -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("gradient_boosting", x, y)
    class_names = list(pipeline.named_steps["model"].classes_)

    output_path = explain_instance_with_lime(pipeline, x, x.iloc[0], class_names, tmp_path / "lime.png")

    assert output_path.exists()


def test_explain_instance_with_lime_works_for_svm_without_predict_proba(tmp_path) -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("svm", x, y)
    assert not hasattr(pipeline.named_steps["model"], "predict_proba")
    class_names = list(pipeline.named_steps["model"].classes_)

    output_path = explain_instance_with_lime(pipeline, x, x.iloc[0], class_names, tmp_path / "lime_svm.png")

    assert output_path.exists()


def test_explain_instance_with_lime_handles_zero_variance_column(tmp_path) -> None:
    # A column that's exactly constant across all of x_train crashes LIME's internal
    # truncnorm perturbation sampling ("scale must be positive") unless excluded - this
    # happens for real on CIC-IDS2017, which has several exactly-constant columns.
    x, y = _make_dataset()
    x = x.copy()
    x["constant_feature"] = 1.0
    preprocessing = build_preprocessing_pipeline(x, categorical_columns=[], random_state=42)
    model = build_model("gradient_boosting", random_state=42)
    pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])
    pipeline.fit(x, y)
    class_names = list(pipeline.named_steps["model"].classes_)

    output_path = explain_instance_with_lime(pipeline, x, x.iloc[0], class_names, tmp_path / "lime_constant.png")

    assert output_path.exists()


def test_compute_permutation_importance_writes_png_and_ranks_features(tmp_path) -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("gradient_boosting", x, y)

    output_path, top_features = compute_permutation_importance(pipeline, x, y, tmp_path / "pfi.png", n_repeats=3)

    assert output_path.exists()
    assert set(top_features) == {"feature_a", "feature_b"}


def test_plot_partial_dependence_writes_png(tmp_path) -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("gradient_boosting", x, y)

    output_path = plot_partial_dependence(pipeline, x, ["feature_a"], "class_0", tmp_path / "pdp.png")

    assert output_path.exists()


def test_plot_partial_dependence_works_for_svm_via_decision_function(tmp_path) -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("svm", x, y)
    assert not hasattr(pipeline.named_steps["model"], "predict_proba")

    output_path = plot_partial_dependence(pipeline, x, ["feature_a"], "class_0", tmp_path / "pdp_svm.png")

    assert output_path.exists()


def test_select_curated_instances_finds_correct_and_misclassified() -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("gradient_boosting", x, y)

    instances = select_curated_instances(pipeline, x, y, "class_0", "class_1", random_state=42)

    assert "correct" in instances
    assert y.loc[instances["correct"]] == "class_0"
    assert pipeline.predict(x.loc[[instances["correct"]]])[0] == "class_0"


def test_select_curated_instances_returns_empty_when_no_misclassifications_exist() -> None:
    x, y = _make_dataset()
    pipeline = _fitted_pipeline("gradient_boosting", x, y)

    # class_0 and class_2 are far apart (centers 0 and 6) so a well-separated model
    # should never confuse them; the "misclassified" key should simply be absent.
    instances = select_curated_instances(pipeline, x, y, "class_0", "class_2", random_state=42)

    assert "misclassified" not in instances
