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
from sklearn.inspection import PartialDependenceDisplay, permutation_importance  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

# Models with a fast, exact structural SHAP explainer (tree ensembles). logistic_regression
# gets LinearExplainer (uses the linear coefficients directly). Anything else
# (neural_network, svm) falls back to a model-agnostic permutation explainer, which is far
# slower since it makes many repeated calls to the model rather than reading its structure
# directly.
_TREE_MODELS = ("gradient_boosting", "random_forest")


def predict_proba_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict_proba-like callable for any classifier, even ones without one.

    No model currently in the catalog actually lacks predict_proba (LinearSVC, the one
    that did, was dropped in favor of logistic regression - see models.py), but this stays
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
    the whole pipeline rather than the bare model. Returns (shap_values, base_values,
    display_data, feature_names):
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
    elif model_name == "logistic_regression":
        explainer = shap.LinearExplainer(model, background_transformed)
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
    """TreeExplainer/LinearExplainer collapse a binary classifier's output to a single column
    (the positive class, model.classes_[1]) rather than one per class, since the other class's
    contribution is just its negation - unlike the multiclass case, or the permutation
    explainer, which always gets one column per class from predict_proba regardless of class
    count. Expand back to a uniform (..., n_classes) shape so downstream code (which assumes
    one SHAP value per class) doesn't need two code paths. The sign convention (index 1 as-is,
    index 0 negated) was verified empirically against the model's own predict_proba output,
    not assumed.
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
) -> Path:
    """Global bar plot of mean |SHAP value| per feature, stacked across all classes."""
    shap.summary_plot(
        [shap_values[:, :, class_idx] for class_idx in range(shap_values.shape[2])],
        x_transformed,
        feature_names=feature_names,
        class_names=class_names,
        plot_type="bar",
        show=False,
    )
    return _save_current_figure(output_path)


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    x_transformed: np.ndarray,
    feature_names: list[str],
    class_index: int,
    output_path: str | Path,
) -> Path:
    """Global beeswarm plot for a single class's SHAP values."""
    plt.figure()
    shap.summary_plot(shap_values[:, :, class_index], x_transformed, feature_names=feature_names, show=False)
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


_BINARY_CLASS_NAMES = {"Attack", "Benign"}


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
    """Local LIME explanation for one instance.

    LimeTabularExplainer requires categorical columns to already be label-encoded as
    integers (categorical_names then supplies the display strings); the pipeline itself
    still expects the original values, so encoding/decoding happens transparently here
    rather than needing the caller to worry about it.

    LIME's default quantile-based discretization of continuous features crashes outright
    (scipy truncnorm "scale must be positive") on the kind of heavily-skewed distributions
    common in these datasets (e.g. CIC-IDS2017's flag counts, ~99% zero, confirmed still
    reproducible after the binary-classification switch) - bin edges collapse regardless of
    cardinality. discretize_continuous defaults to False (sidesteps the discretizer entirely,
    perturbing continuous features via a Gaussian around each feature's std instead) and
    should only be set True for a dataset verified not to hit this crash (confirmed safe for
    UNSW-NB15 specifically).

    For the two binary (Attack/Benign) datasets, the explanation is always computed relative
    to the Attack class specifically - the security-critical one - regardless of which class
    the model actually predicted for this instance, so a correctly-flagged attack and a missed
    one can be read on the same footing. Rendered with a fixed Class 0 (Benign, blue) / Class 1
    (Attack, orange) layout; true multiclass datasets (e.g. DAPT2020) fall back to LIME's own
    default single-panel rendering for whichever class was actually predicted, since a fixed
    two-sided layout doesn't generalize past two classes.
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

    if set(class_names) == _BINARY_CLASS_NAMES:
        attack_index = class_names.index("Attack")
        explanation = explainer.explain_instance(
            instance_encoded.to_numpy(dtype=float), _predict_fn_on_encoded, num_features=num_features,
            labels=(attack_index,),
        )
        fig = _plot_lime_binary_explanation(explanation, class_names, instance, columns)
    else:
        explanation = explainer.explain_instance(
            instance_encoded.to_numpy(dtype=float), _predict_fn_on_encoded, num_features=num_features, top_labels=1
        )
        label = explanation.available_labels()[0]
        fig = explanation.as_pyplot_figure(label=label)

    return _save_figure(fig, output_path)


def _feature_name_from_condition(condition: str, known_columns: list[str]) -> str:
    """Recover the underlying feature name from a LIME condition string, whatever form it
    takes ("sttl", "sttl <= 0.14", "0.00 < ct_srv_dst <= 1.00", "service=unknown") - matched
    against the known column list rather than parsed by operator, since that's robust to any
    of LIME's condition formats. Longest match wins, so one column name being a substring of
    another (e.g. "ct_srv_dst" inside "ct_srv_dst_ltm") doesn't pick the wrong one.
    """
    matches = [column for column in known_columns if column in condition]
    return max(matches, key=len) if matches else condition


def _plot_lime_binary_explanation(
    explanation: object, class_names: list[str], instance: pd.Series, known_columns: list[str]
) -> plt.Figure:
    """Render a two-class LIME explanation with a fixed Class 0 (Benign, blue) / Class 1
    (Attack, orange) layout: a small prediction-probability inset in the top-left, a diverging
    feature-contribution chart in the middle, and a matching feature-value table on the right.

    The explanation is always for the Attack class (see explain_instance_with_lime), so
    positive weights support Attack (orange, right) and negative weights support Benign
    (blue, left) consistently, regardless of which class was actually predicted.
    """
    blue, orange = "#1f77b4", "#ff7f0e"
    attack_index = class_names.index("Attack")
    benign_index = class_names.index("Benign")
    proba = explanation.predict_proba
    contributions = explanation.as_list(label=attack_index)

    conditions = [condition for condition, _ in contributions]
    weights = [weight for _, weight in contributions]
    colors = [orange if weight > 0 else blue for weight in weights]

    fig = plt.figure(figsize=(13, max(3.5, len(contributions) * 0.42 + 1.2)), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.1, 2.2, 1.6], height_ratios=[1, 2.2])

    proba_ax = fig.add_subplot(grid[0, 0])
    proba_ax.barh([0, 1], [proba[benign_index], proba[attack_index]], color=[blue, orange])
    proba_ax.set_yticks([0, 1])
    proba_ax.set_yticklabels(["Class 0", "Class 1"], fontsize=8)
    proba_ax.invert_yaxis()
    proba_ax.set_xlim(0, 1)
    proba_ax.tick_params(axis="x", labelsize=7)
    proba_ax.set_title("Prediction probabilities", fontsize=9)
    for position, probability in enumerate([proba[benign_index], proba[attack_index]]):
        proba_ax.text(min(probability + 0.04, 0.85), position, f"{probability:.2f}", va="center", fontsize=8)

    contrib_ax = fig.add_subplot(grid[:, 1])
    y_pos = range(len(conditions))
    contrib_ax.barh(y_pos, weights, color=colors)
    contrib_ax.set_yticks(list(y_pos))
    contrib_ax.set_yticklabels(conditions)
    contrib_ax.invert_yaxis()
    contrib_ax.axvline(0, color="black", linewidth=0.8)
    contrib_ax.text(0.0, 1.02, "Class 0", color=blue, ha="left", fontsize=12, transform=contrib_ax.transAxes)
    contrib_ax.text(1.0, 1.02, "Class 1", color=orange, ha="right", fontsize=12, transform=contrib_ax.transAxes)

    table_ax = fig.add_subplot(grid[:, 2])
    table_ax.axis("off")
    row_names, row_values, row_colors = [], [], []
    for condition, weight in contributions:
        feature_name = _feature_name_from_condition(condition, known_columns)
        value = instance.get(feature_name, "")
        row_names.append(feature_name)
        if isinstance(value, (int, float, np.number)):
            row_values.append(f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}")
        else:
            row_values.append(str(value))
        row_colors.append(orange if weight > 0 else blue)

    table = table_ax.table(
        cellText=list(zip(row_names, row_values, strict=True)),
        colLabels=["Feature", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.65, 0.35],
    )
    for row_index, color in enumerate(row_colors, start=1):
        for column_index in (0, 1):
            cell = table[row_index, column_index]
            cell.set_facecolor(color)
            cell.get_text().set_color("white")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    return fig


def compute_permutation_importance(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    output_path: str | Path,
    *,
    scoring: str = "f1_macro",
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 15,
) -> tuple[Path, list[str]]:
    """Permutation feature importance on the full pipeline (raw, pre-encoding feature names).

    Returns the plot path and the top_n most important feature names, for PDP to reuse.
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


def plot_partial_dependence(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    features: list[str],
    target_class: str,
    output_path: str | Path,
    *,
    random_state: int = 42,
) -> Path:
    """Partial dependence plots for a handful of features, for one target class.

    Numeric columns are cast to float first: sklearn's PDP rejects integer dtypes outright.
    response_method="auto" falls back to decision_function for models without predict_proba.

    kind="both" overlays individual conditional expectation (ICE) curves - one line per
    sampled instance - underneath the usual average PDP line. The average alone can look
    flat or negligible even when a feature has a real but heterogeneous effect (pushing some
    instances toward the target class and others away from it); that distinction matters
    here specifically, since "this feature matters, but only for some traffic" is a
    materially different finding than "this feature doesn't matter" for a thesis about
    explanations informing security decisions. subsample caps how many individual lines are
    drawn - plotting all of x_data would overplot into an unreadable smear - and random_state
    makes which instances get drawn reproducible, matching every other sampling step in this
    pipeline.
    """
    x_data = x_data.copy()
    numeric_columns = x_data.select_dtypes(include="number").columns
    x_data[numeric_columns] = x_data[numeric_columns].astype(float)

    display = PartialDependenceDisplay.from_estimator(
        pipeline,
        x_data,
        features=features,
        target=target_class,
        response_method="auto",
        kind="both",
        subsample=50,
        random_state=random_state,
    )
    n_cols = min(3, len(features))
    n_rows = -(-len(features) // n_cols)
    display.figure_.set_size_inches(4.5 * n_cols, 3.5 * n_rows)
    display.figure_.tight_layout()
    return _save_figure(display.figure_, output_path)


def select_curated_instances(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    target_class: str,
    confused_with_class: str,
    *,
    random_state: int = 42,
) -> dict[str, int]:
    """Pick one correctly-classified and one misclassified instance of target_class.

    Instance selection informed by confusion-matrix findings rather than random sampling,
    e.g. a genuine target_class case the model got right, and one it mistook for
    confused_with_class specifically - ties local explanations back to known error patterns.
    """
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
