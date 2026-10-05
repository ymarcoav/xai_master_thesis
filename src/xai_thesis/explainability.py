"""SHAP, LIME and permutation feature importance for the fitted model pipelines, and their plots."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

# Non-interactive backend: figures are only saved to disk.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

# Models explained with the exact tree-specific SHAP explainer. The others (neural_network, svm)
# use the model-agnostic permutation explainer, which estimates Shapley values by sampling.
_TREE_MODELS = ("random_forest", "xgboost")

# Number of permutations for the permutation explainer. SHAP's default budget (max_evals=500)
# amounts to only about 4 permutations at 52-59 features, a high-variance estimate; 50 gives a
# much more stable one. The run time grows roughly linearly with this number.
_SHAP_PERMUTATION_MULTIPLIER = 50


def predict_proba_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return the model's predict_proba, or softmax(decision_function) for a classifier
    without one. All four models in the catalog provide predict_proba.
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
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Compute SHAP values for x_data using the fastest suitable explainer.

    The explainers operate on the fitted model, i.e. in the transformed (scaled, one-hot encoded)
    feature space. For reporting, the attributions of a categorical feature's one-hot columns are
    summed back into that feature, and the display values are the original, untransformed ones.

    Output space by model:
      - random_forest: TreeExplainer (tree_path_dependent, no background), probabilities.
      - xgboost: TreeExplainer, log-odds. SHAP supports probability output for XGBoost only for
        binary objectives, so for these multiclass models the values stay in log-odds; their
        level is not comparable with the other models', their ranking and signs are.
      - svm, neural_network: permutation explainer over predict_proba, probabilities, with
        background_data as the reference distribution.

    The permutation explainer is seeded with random_state and runs a fixed number of
    permutations (_SHAP_PERMUTATION_MULTIPLIER). The tree explainer is exact and deterministic.

    Returns (shap_values, base_values, display_data, feature_names):
      - shap_values: shape (n_samples, n_original_features, n_classes), already collapsed
      - base_values: shape (n_samples, n_classes); the tree explainer's per-class expected
        value is broadcast across samples.
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
        # The background is used as passed (a small class-balanced sample, see pipeline.py).
        max_evals = _SHAP_PERMUTATION_MULTIPLIER * (2 * x_transformed.shape[1] + 1)
        explainer = shap.Explainer(
            predict_proba_fn(model), background_transformed, algorithm="permutation", seed=random_state
        )
        explanation = explainer(x_transformed, max_evals=max_evals)
        shap_values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)

    shap_values, display_data, feature_names = _collapse_to_original_features(
        shap_values, feature_names, x_data, categorical_columns
    )
    return shap_values, base_values, display_data, feature_names


def _expand_binary_class_dimension(
    shap_values: np.ndarray, expected_value: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Give a binary margin-space explanation one column per class.

    For a binary problem, TreeExplainer can return a single log-odds column for the positive
    class. That column is kept for class 1 and negated for class 0. Multiclass output, and
    output that is already per class, is returned unchanged.
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
    """Sum the SHAP values of each categorical feature's one-hot columns into that feature, and
    replace the transformed display values with the original ones.

    Numeric features map one-to-one onto a transformed column and pass through unchanged. The
    one-hot columns are summed with their signs, which gives the categorical feature's net
    contribution to the prediction; its global importance is therefore the mean |net
    contribution|.
    """
    # Longest name first, so that a column name which prefixes another cannot claim its dummies.
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


def class_colors(class_names: Sequence[str]) -> list[tuple[float, float, float, float]]:
    """One fixed colour per class, indexed like class_names (model.classes_ order).

    Shared by plot_shap_bar and compute_permutation_importance, so that a class has the same
    colour in both global figures. Uses SHAP's default red_blue_circle palette.
    """
    return [shap.plots.colors.red_blue_circle(i / len(class_names)) for i in range(len(class_names))]


def plot_shap_bar(
    shap_values: np.ndarray,
    x_transformed: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    output_path: str | Path,
    max_display: int = 10,
) -> Path:
    """Global bar plot of mean |SHAP value| per feature, stacked across all classes.

    Shows the top max_display features, like the other explanation plots. Each bar's segments
    are the per-class mean |SHAP value|. The x-axis label is shortened because SHAP's default
    label is cut off at this figure width. Segments are coloured by class (class_colors), so the
    colours match the permutation importance plot; the stack order is SHAP's default.
    """
    # SHAP's default stack order (classes by overall mean |SHAP|), coloured by class.
    class_inds = np.argsort(-np.abs(shap_values).mean(axis=(0, 1)))
    palette = class_colors(class_names)
    shap.summary_plot(
        [shap_values[:, :, class_idx] for class_idx in range(shap_values.shape[2])],
        x_transformed,
        feature_names=feature_names,
        class_names=class_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
        class_inds=list(class_inds),
        color=lambda stack_index: palette[class_inds[stack_index]],
    )
    plt.gca().set_xlabel("mean(|SHAP value|)", fontsize=13)
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
    *,
    value_label: str = "probability",
) -> Path:
    """Local waterfall plot explaining one instance's prediction for one class.

    The x-axis runs from the base value E[f(X)] to the instance's output f(x); each bar is one
    feature's signed contribution. value_label names the output space shown on the x-axis:
    "probability", or "log-odds" for XGBoost (see compute_shap_values).
    """
    explanation = shap.Explanation(
        values=shap_values[instance_index, :, class_index],
        base_values=base_values[instance_index, class_index],
        data=x_transformed_instance,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False, max_display=max_display)
    # The waterfall plot adds a secondary top axis and makes it current, so the label is set on
    # the bottom axis explicitly; labelpad clears the "E[f(X)] = " annotation below it.
    plt.gcf().axes[0].set_xlabel(f"Model output: {value_label}", fontsize=12, labelpad=28)
    return _save_current_figure(output_path)


def _build_lime_explainer(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    class_names: list[str],
    *,
    categorical_columns: Sequence[str] = (),
    random_state: int = 42,
    discretize_continuous: bool = True,
) -> tuple[LimeTabularExplainer, Callable[[np.ndarray], np.ndarray], Callable[[pd.Series], np.ndarray], list[str]]:
    """Build a LimeTabularExplainer and the matching predict function for a fitted pipeline.

    Returns (explainer, predict_fn_on_encoded, encode_instance, columns): `encode_instance`
    turns one raw test row into the vector explain_instance expects, and `columns` is the
    (zero-variance-dropped) feature order LIME's local_exp indices refer to.

    Building the explainer scans all of x_train, so callers that explain many instances build it
    once.

    LimeTabularExplainer expects categorical columns encoded as integers (categorical_names holds
    the display strings), while the pipeline expects the original values; the returned
    functions translate between the two.

    discretize_continuous=True (LIME's default) bins continuous features into training quartiles,
    which gives the plots readable range labels. The discretizer requires NaN-free data, so
    numeric NaNs in the training data and in each explained row are filled with the training
    medians, which is exactly what the pipeline's median imputer does; the model's prediction for
    a row is therefore unchanged.
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
    fill_columns = [column for column in columns if column in numeric_columns and column not in categorical_columns]
    fill_values = x_train_encoded[fill_columns].median()
    x_train_encoded[fill_columns] = x_train_encoded[fill_columns].fillna(fill_values)

    def _encode_instance(instance: pd.Series) -> np.ndarray:
        instance_encoded = instance[columns].copy()
        for column, mapping in category_to_code.items():
            instance_encoded[column] = mapping[str(instance[column])]
        return instance_encoded.fillna(fill_values).to_numpy(dtype=float)

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
    return explainer, _predict_fn_on_encoded, _encode_instance, columns


def _build_lime_explanation(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    instance: pd.Series,
    class_names: list[str],
    *,
    categorical_columns: Sequence[str] = (),
    num_features: int = 8,
    random_state: int = 42,
    discretize_continuous: bool = True,
    class_index: int | None = None,
) -> tuple[Any, int, list[str]]:
    """Build the explainer and run explain_instance for a single row. Returns
    (explanation, predicted_label, columns).

    class_index, if given, fixes the explained class (LIME's labels argument) instead of the
    default top_labels=1, the most probable class. The interview plots pass the predicted class,
    so that LIME explains the same class as the SHAP waterfall; for the SVM, whose prediction
    uses one-vs-one voting, the predicted class is not always the most probable one.
    """
    explainer, predict_fn_on_encoded, encode_instance, columns = _build_lime_explainer(
        pipeline,
        x_train,
        class_names,
        categorical_columns=categorical_columns,
        random_state=random_state,
        discretize_continuous=discretize_continuous,
    )
    explain_kwargs = {"labels": (class_index,)} if class_index is not None else {"top_labels": 1}
    explanation = explainer.explain_instance(
        encode_instance(instance),
        predict_fn_on_encoded,
        num_features=num_features,
        **explain_kwargs,
    )
    return explanation, explanation.available_labels()[0], columns


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
    discretize_continuous: bool = True,
    class_index: int | None = None,
) -> Path:
    """Local LIME explanation for one instance as LIME's feature-weight bar chart, with the
    default title removed and an x-axis label added. See _build_lime_explanation for class_index.
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
        class_index=class_index,
    )
    fig = explanation.as_pyplot_figure(label=label)
    ax = fig.axes[0]
    ax.set_title("")
    ax.set_xlabel("Feature weight")

    return _save_figure(fig, output_path)


def compute_permutation_importance(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    output_path: str | Path,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 10,
) -> tuple[Path, list[str], Any]:
    """Permutation feature importance on the full pipeline (raw feature names), measured as each
    feature's drop in macro-F1 over n_repeats shuffles, and plotted stacked by class.

    Macro-F1 is the mean of the K per-class F1 scores, so a feature's macro-F1 drop splits
    exactly into K per-class segments (each class's F1 drop divided by K). All scores come from
    one multi-metric permutation_importance call. A class's segment can be negative (shuffling
    slightly helped that class); negative segments are drawn left of zero. Classes are stacked
    and coloured as in plot_shap_bar.

    Returns the plot path, the top_n feature names, and sklearn's result for macro-F1
    (importances_mean, importances_std, importances).
    """
    class_names = [str(name) for name in pipeline.named_steps["model"].classes_]
    labels = list(pipeline.named_steps["model"].classes_)

    def _scorer(estimator: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        per_class = f1_score(y, estimator.predict(x), labels=labels, average=None, zero_division=0)
        scores = {f"f1::{name}": float(score) for name, score in zip(class_names, per_class)}
        return {"f1_macro": float(per_class.mean()), **scores}

    results = permutation_importance(
        pipeline, x_data, y_data, scoring=_scorer, n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    result = results["f1_macro"]
    # (n_features, n_classes): each class's share of the macro-F1 drop.
    per_class_drops = [results[f"f1::{name}"].importances_mean for name in class_names]
    contributions = np.column_stack(per_class_drops) / len(class_names)

    order = np.argsort(result.importances_mean)[::-1][:top_n]
    top_features = [x_data.columns[i] for i in order]
    plot_path = _plot_stacked_permutation_importance(
        contributions[order],
        result.importances_mean[order],
        result.importances_std[order],
        top_features,
        class_names,
        output_path,
    )

    return plot_path, top_features, result


def _plot_stacked_permutation_importance(
    contributions: np.ndarray,
    totals: np.ndarray,
    totals_std: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    output_path: str | Path,
) -> Path:
    """Draw compute_permutation_importance's per-class stacked bars, most important on top.

    contributions is (n_features, n_classes) in display order. Positive segments stack to the
    right of zero and negative ones to the left; a black diamond with an error bar (standard
    deviation over repeats) marks each feature's net macro-F1 drop.
    """
    n_features = contributions.shape[0]
    y_pos = np.arange(n_features)[::-1]
    class_order = np.argsort(-contributions.sum(axis=0))
    palette = class_colors(class_names)

    fig, ax = plt.subplots(figsize=(8, n_features * 0.4 + 1.5))
    right = np.zeros(n_features)
    left = np.zeros(n_features)
    for class_index in class_order:
        values = contributions[:, class_index]
        positive, negative = np.clip(values, 0, None), np.clip(values, None, 0)
        color = palette[class_index]
        ax.barh(y_pos, positive, 0.7, left=right, color=color, label=class_names[class_index])
        ax.barh(y_pos, negative, 0.7, left=left, color=color)
        right += positive
        left += negative

    ax.errorbar(
        totals, y_pos, xerr=totals_std, fmt="D", color="#222222", markersize=4, elinewidth=1, capsize=3
    )
    if left.min() < 0:
        ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_yticks(y_pos, feature_names, fontsize=13)
    ax.set_xlabel("Permutation importance (macro-F1 drop)", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    return _save_figure(fig, output_path)


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
    """Pick one correctly classified instance of target_class and one misclassified as
    confused_with_class, at random with a fixed seed. Pass predictions (aligned to x_test.index)
    to reuse an existing prediction of the whole test set.
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
