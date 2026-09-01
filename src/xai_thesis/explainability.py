from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

# Force the non-interactive backend: this module only saves figures to disk, and the
# default backend can try to initialize a Tk display that isn't available headlessly.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

# Models with a fast, exact structural SHAP explainer (tree ensembles). Everything else
# (neural_network, svm) falls back to a model-agnostic permutation explainer, which is far
# slower since it makes many repeated calls to the model rather than reading its structure
# directly. xgboost is one of TreeExplainer's best-supported cases (verified directly against
# the installed shap/xgboost versions) - the label-encoding subclass in models.py doesn't
# interfere, since it's a real XGBClassifier subclass, not a wrapper.
#
# gradient_boosting (also TreeExplainer) and logistic_regression (its own dedicated
# shap.LinearExplainer branch, since removed) were both removed from the model catalog
# entirely on 2026-08-24 (explicit user request) - see models.py for the full removal note.
_TREE_MODELS = ("random_forest", "xgboost")


def predict_proba_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict_proba-like callable for any classifier, even ones without one.

    No model currently in the catalog actually lacks predict_proba (LinearSVC, the one that
    did, was dropped early on - see models.py), but this stays
    generic rather than assuming that always holds: softmax(decision_function) is a
    deterministic, argmax-preserving stand-in for a margin-only classifier (the predicted
    class is always identical to model.predict()), unlike CalibratedClassifierCV, which was
    found to shift predictions under severe class imbalance and is deliberately not used.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba

    def _softmax_decision_function(x: np.ndarray) -> np.ndarray:
        scores = model.decision_function(x)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    return _softmax_decision_function


def clean_feature_names(feature_names: list[str]) -> list[str]:
    """Strip the ColumnTransformer's "numeric__"/"categorical__" prefixes for display."""
    return [name.split("__", 1)[-1] for name in feature_names]


def compute_shap_values(
    pipeline: Pipeline,
    model_name: str,
    x_data: pd.DataFrame,
    background_data: pd.DataFrame,
    *,
    categorical_columns: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Compute SHAP values for x_data using the fastest suitable explainer.

    TreeExplainer and LinearExplainer need direct access to the fitted model, which only ever
    sees the transformed (quantile-scaled, one-hot-encoded) feature space, so the underlying
    Shapley values are computed there. For display, one-hot dummy columns' attributions are
    summed back into their parent categorical feature, and the transformed numeric/categorical
    values are swapped for the original, untransformed ones - unlike LIME/permutation
    importance/PDP, which already operate on the original feature space directly by wrapping
    the whole pipeline rather than the bare model.

    Base-value footing differs by explainer and is not fully unifiable here: the tree path
    (random_forest, xgboost) runs TreeExplainer with feature_perturbation="tree_path_dependent"
    (no background - xgboost's SHAP build rejects the interventional+background path outright),
    so its base value is the model's own class_weight/sample_weight-adjusted internal prior,
    which for this project's balanced weighting sits near 1/n_classes. The permutation path
    (svm, neural_network) uses background_data directly; pipeline.py passes a class-balanced
    sample there so its base value lands on that same near-uniform footing rather than on the
    85-96%-Benign raw prior. xgboost's base value is in margin space and still not directly
    comparable in level to the others (attribution directions are).

    Returns (shap_values, base_values, display_data, feature_names):
      - shap_values: shape (n_samples, n_original_features, n_classes), already collapsed
      - base_values: shape (n_samples, n_classes). TreeExplainer/LinearExplainer give one
        constant expected_value per class (broadcast across samples here for a uniform
        shape); the permutation explainer (no .expected_value attribute) already returns
        base values per instance directly.
      - display_data: shape (n_samples, n_original_features), original units/categories,
        object-dtype since numeric and categorical columns are mixed
    """
    preprocessing = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    feature_names = clean_feature_names(list(preprocessing.get_feature_names_out()))
    n_classes = len(model.classes_)

    x_transformed = preprocessing.transform(x_data)
    background_transformed = preprocessing.transform(background_data)
    n_samples = x_transformed.shape[0]

    if model_name in _TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(x_transformed))
        expected_value = np.atleast_1d(np.asarray(explainer.expected_value))
        shap_values, expected_value = _expand_binary_class_dimension(shap_values, expected_value, n_classes)
        base_values = np.broadcast_to(expected_value, (n_samples, expected_value.shape[0]))
    else:
        background_summary = shap.sample(background_transformed, min(50, len(background_transformed)))
        explainer = shap.Explainer(predict_proba_fn(model), background_summary, algorithm="permutation")
        explanation = explainer(x_transformed)
        shap_values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)

    shap_values, display_data, feature_names = _collapse_to_original_features(
        shap_values, feature_names, x_data, categorical_columns
    )
    return shap_values, base_values, display_data, feature_names


def _expand_binary_class_dimension(
    shap_values: np.ndarray, expected_value: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Some binary classifiers' TreeExplainer/LinearExplainer output collapses to a single raw
    margin/log-odds column (the positive class, model.classes_[1]) rather than one per class,
    since the other class's contribution there is just its negation - unlike the multiclass
    case, the permutation explainer (always one column per class from predict_proba regardless
    of class count), or - verified directly, not assumed - RandomForestClassifier's own
    TreeExplainer output, which is already 3D/per-class for a binary problem (expected_value
    already something like [0.4996, 0.5004], summing to ~1 like real complementary
    probabilities, not [-x, x]). The ndim==3 check below is what tells these two cases apart;
    the sign convention this function applies (index 1 as-is, index 0 negated) only holds for
    the collapsed, margin-space case and was verified empirically against that model's own
    predict_proba output, not assumed - don't assume it generalizes to every future model
    without checking shap_values.ndim / expected_value's own shape first, the way random_forest
    already disproved it would.
    """
    if n_classes != 2 or shap_values.ndim == 3:
        return shap_values, expected_value
    return np.stack([-shap_values, shap_values], axis=-1), np.array([-expected_value[0], expected_value[0]])


def _collapse_to_original_features(
    shap_values: np.ndarray,
    transformed_feature_names: list[str],
    x_data: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sum one-hot dummy columns' SHAP values back into their parent categorical feature, and
    replace the (transformed-space) display values with the original, untransformed ones.

    Numeric features map 1:1 onto a transformed column, so their SHAP values pass through
    unchanged; only their displayed value changes (original units instead of quantile-scaled).
    """
    # Longest-name-first so a categorical column name that happens to prefix another (e.g. a
    # column "a" and a column "a_b") doesn't grab the wrong dummy block.
    sorted_categorical = sorted(categorical_columns, key=len, reverse=True)

    def _owner(name: str) -> str:
        if name not in categorical_columns and name in x_data.columns:
            return name
        for column in sorted_categorical:
            if name == column or name.startswith(f"{column}_"):
                return column
        raise ValueError(f"Could not map transformed feature {name!r} back to an original column")

    owners = [_owner(name) for name in transformed_feature_names]
    original_columns = list(dict.fromkeys(owners))  # first-seen order, de-duplicated

    n_samples, _, n_classes = shap_values.shape
    collapsed_shap = np.zeros((n_samples, len(original_columns), n_classes))
    for transformed_index, owner in enumerate(owners):
        collapsed_shap[:, original_columns.index(owner), :] += shap_values[:, transformed_index, :]

    display_data = x_data[original_columns].to_numpy(dtype=object)
    return collapsed_shap, display_data, original_columns


def plot_shap_bar(
    shap_values: np.ndarray,
    x_transformed: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    output_path: str | Path,
    max_display: int = 10,
) -> Path:
    """Global bar plot of mean |SHAP value| per feature, stacked across all classes.

    max_display defaults to 10 to match every other explanation plot in this project (SHAP
    waterfall's own max_display, and - as of 2026-08-29 - plot_shap_beeswarm/
    compute_permutation_importance below), not shap.summary_plot's own default. Left
    unspecified, summary_plot silently defaults max_display to 7 here specifically (the
    multi-class/list-of-arrays code path this function's plot_type="bar" call takes) but 20 for
    plot_shap_beeswarm below (the single-array path) - two "global" plots in this project's own
    suite showing a different number of features from each other by accident, not by design.
    """
    shap.summary_plot(
        [shap_values[:, :, class_idx] for class_idx in range(shap_values.shape[2])],
        x_transformed,
        feature_names=feature_names,
        class_names=class_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    return _save_current_figure(output_path)


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    x_transformed: np.ndarray,
    feature_names: list[str],
    class_index: int,
    output_path: str | Path,
    max_display: int = 10,
) -> Path:
    """Global beeswarm plot for a single class's SHAP values.

    max_display defaults to 10 - see plot_shap_bar's docstring for why this needed to be set
    explicitly rather than left to shap.summary_plot's own (here, different) internal default.
    """
    plt.figure()
    shap.summary_plot(
        shap_values[:, :, class_index],
        x_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    return _save_current_figure(output_path)


def plot_shap_waterfall(
    shap_values: np.ndarray,
    base_values: np.ndarray,
    x_transformed_instance: np.ndarray,
    feature_names: list[str],
    instance_index: int,
    class_index: int,
    output_path: str | Path,
    max_display: int = 10,
) -> Path:
    """Local waterfall plot explaining one instance's prediction for one class."""
    explanation = shap.Explanation(
        values=shap_values[instance_index, :, class_index],
        base_values=base_values[instance_index, class_index],
        data=x_transformed_instance,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False, max_display=max_display)
    return _save_current_figure(output_path)


def _build_lime_explanation(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    instance: pd.Series,
    class_names: list[str],
    *,
    categorical_columns: Sequence[str] = (),
    num_features: int = 8,
    random_state: int = 42,
    discretize_continuous: bool = False,
) -> tuple[Any, int, list[str]]:
    """Shared setup for LIME explanations: encodes categoricals, builds the explainer, and
    runs explain_instance. Returns (explanation, predicted_label, columns), where columns is
    the (zero-variance-dropped) feature order that local_exp's feature indices refer to.

    Factored out of explain_instance_with_lime so the notebook-style single-panel bar chart
    and the three-panel Figure-2-style layout can share one explanation run instead of
    computing it twice.

    LimeTabularExplainer requires categorical columns to already be label-encoded as
    integers (categorical_names then supplies the display strings); the pipeline itself
    still expects the original values, so encoding/decoding happens transparently here
    rather than needing the caller to worry about it.

    LIME's default quantile-based discretization of continuous features crashes outright
    (scipy truncnorm "scale must be positive") on the kind of heavily-skewed distributions
    common in these datasets (e.g. CIC-IDS2017's flag counts, ~99% zero) - bin edges collapse
    regardless of cardinality. discretize_continuous defaults to False (sidesteps the
    discretizer entirely, perturbing continuous features via a Gaussian around each feature's
    std instead) and should only be set True for a dataset verified not to hit this crash
    (confirmed safe for UNSW-NB15 specifically).

    top_labels=1 explains whichever class was actually predicted - every dataset in this
    project's catalog is multiclass, so there's no fixed two-class layout to special-case.
    """
    all_columns = list(x_train.columns)
    numeric_columns = x_train.select_dtypes(include="number").columns
    zero_variance_columns = [column for column in numeric_columns if x_train[column].std() == 0]
    constant_values = {column: x_train[column].iloc[0] for column in zero_variance_columns}

    columns = [column for column in all_columns if column not in zero_variance_columns]
    categorical_indices = [columns.index(column) for column in categorical_columns]
    category_values = {column: sorted(x_train[column].astype(str).unique()) for column in categorical_columns}
    category_to_code = {
        column: {value: code for code, value in enumerate(values)} for column, values in category_values.items()
    }
    code_to_category = {column: dict(enumerate(values)) for column, values in category_values.items()}

    x_train_encoded = x_train[columns].copy()
    for column in categorical_columns:
        x_train_encoded[column] = x_train_encoded[column].astype(str).map(category_to_code[column])

    instance_encoded = instance[columns].copy()
    for column in categorical_columns:
        instance_encoded[column] = category_to_code[column][str(instance[column])]

    model = pipeline.named_steps["model"]
    predict_fn = predict_proba_fn(model)

    def _predict_fn_on_encoded(x_array: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame(x_array, columns=columns)
        for column in categorical_columns:
            x_df[column] = x_df[column].round().astype(int).map(code_to_category[column])
        for column, value in constant_values.items():
            x_df[column] = value
        transformed = pipeline.named_steps["preprocessing"].transform(x_df[all_columns])
        return predict_fn(transformed)

    explainer = LimeTabularExplainer(
        x_train_encoded.to_numpy(dtype=float),
        feature_names=columns,
        class_names=class_names,
        categorical_features=categorical_indices,
        categorical_names={
            index: category_values[column]
            for index, column in zip(categorical_indices, categorical_columns, strict=True)
        },
        mode="classification",
        random_state=random_state,
        discretize_continuous=discretize_continuous,
    )

    explanation = explainer.explain_instance(
        instance_encoded.to_numpy(dtype=float), _predict_fn_on_encoded, num_features=num_features, top_labels=1
    )
    label = explanation.available_labels()[0]
    return explanation, label, columns


def explain_instance_with_lime(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    instance: pd.Series,
    class_names: list[str],
    output_path: str | Path,
    *,
    categorical_columns: Sequence[str] = (),
    num_features: int = 8,
    random_state: int = 42,
    discretize_continuous: bool = False,
) -> Path:
    """Local LIME explanation for one instance, rendered as LIME's default single-panel
    feature-weight bar chart. See _build_lime_explanation for the encoding/explainer details.
    """
    explanation, label, _columns = _build_lime_explanation(
        pipeline,
        x_train,
        instance,
        class_names,
        categorical_columns=categorical_columns,
        num_features=num_features,
        random_state=random_state,
        discretize_continuous=discretize_continuous,
    )
    fig = explanation.as_pyplot_figure(label=label)

    return _save_figure(fig, output_path)


def compute_permutation_importance(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    output_path: str | Path,
    *,
    scoring: str = "f1_macro",
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 10,
) -> tuple[Path, list[str]]:
    """Permutation feature importance on the full pipeline (raw, pre-encoding feature names).

    top_n defaults to 10 (changed from 15 on 2026-08-29) to match every other explanation plot
    in this project - see plot_shap_bar's docstring for the full story of why these needed
    aligning. Returns the plot path and the top_n most important feature names; the latter is
    currently unused by pipeline.py's own caller (originally consumed by plot_partial_dependence,
    removed 2026-08-28) but kept in the return signature since it's a real, free-standing return
    value some other caller could still want.
    """
    result = permutation_importance(
        pipeline, x_data, y_data, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    top_features = [x_data.columns[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.35)))
    ax.barh(
        [x_data.columns[i] for i in order][::-1],
        result.importances_mean[order][::-1],
        xerr=result.importances_std[order][::-1],
        color="#4C72B0",
    )
    ax.set_xlabel(f"Permutation importance ({scoring} drop)")
    fig.tight_layout()
    plot_path = _save_figure(fig, output_path)

    return plot_path, top_features


def select_curated_instances(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    target_class: str,
    confused_with_class: str,
    *,
    predictions: pd.Series | None = None,
    random_state: int = 42,
) -> dict[str, int]:
    """Pick one correctly-classified and one misclassified instance of target_class.

    Instance selection informed by confusion-matrix findings rather than random sampling,
    e.g. a genuine target_class case the model got right, and one it mistook for
    confused_with_class specifically - ties local explanations back to known error patterns.
    Pass predictions (aligned to x_test.index) to reuse an already-computed full-test-set
    prediction instead of recomputing it - matters for the slow svm pipeline.
    """
    if predictions is None:
        predictions = pd.Series(pipeline.predict(x_test), index=x_test.index)
    is_target = y_test == target_class

    rng = np.random.default_rng(random_state)
    instances: dict[str, int] = {}

    correct_candidates = x_test.index[is_target & (predictions == target_class)]
    if len(correct_candidates) > 0:
        instances["correct"] = rng.choice(correct_candidates)

    misclassified_candidates = x_test.index[is_target & (predictions == confused_with_class)]
    if len(misclassified_candidates) > 0:
        instances["misclassified"] = rng.choice(misclassified_candidates)

    return instances


def _save_current_figure(output_path: str | Path) -> Path:
    return _save_figure(plt.gcf(), output_path)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
