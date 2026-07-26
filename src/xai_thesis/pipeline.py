from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from xai_thesis.datasets import DATASET_REGISTRY
from xai_thesis.datasets.base import DatasetConfig, load_dataset_config
from xai_thesis.evaluation import evaluate_classifier, plot_confusion_matrix, plot_label_distribution
from xai_thesis.explainability import (
    compute_permutation_importance,
    compute_shap_values,
    explain_instance_with_lime,
    plot_partial_dependence,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_waterfall,
    select_curated_instances,
)
from xai_thesis.models import PARAM_DISTRIBUTIONS, build_model, train_model
from xai_thesis.preprocessing import (
    balance_classes,
    build_preprocessing_pipeline,
    split_by_official_split_column,
    split_chronologically_by_source,
    split_features_target,
    stratified_subsample,
)
from xai_thesis.utils import dump_yaml, ensure_directory, load_yaml

# MLPClassifier has no class_weight/sample_weight support, unlike the other models here
# (see models.py), so it needs its own training-time rebalancing via resampling instead.
# Tried adding gradient_boosting/logistic_regression here too, to test whether resampling
# helps on DAPT2020's most data-starved classes beyond their existing class_weight="balanced"
# - it made both models measurably worse (LR f1_macro 0.352->0.318, GB 0.644->0.601), so
# it's restricted back to just the one model that has no other option.
_MODELS_REQUIRING_RESAMPLING = {"neural_network"}

_METRIC_EXTRACTORS = {
    "accuracy": lambda report: report["accuracy"],
    "precision_macro": lambda report: report["macro avg"]["precision"],
    "recall_macro": lambda report: report["macro avg"]["recall"],
    "f1_macro": lambda report: report["macro avg"]["f1-score"],
}

_SUPPORTED_PREPROCESSING_STRATEGY = "default_tabular"


def preprocess_dataset(config_path: str | Path) -> tuple[Path, Path]:
    """Load, clean, and split one dataset, writing train/test CSVs to its processed_path."""
    config = load_dataset_config(config_path)
    dataset = DATASET_REGISTRY[config.name](config)
    data = dataset.build()

    train, test = _split_dataset(config, data)

    output_dir = ensure_directory(config.processed_path)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def _split_dataset(config: DatasetConfig, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # UNSW-NB15 ships with an official train/test split; the others don't, so they're
    # split chronologically per source file instead (see preprocessing.py for why).
    if config.name == "unsw_nb15":
        return split_by_official_split_column(data, split_column="split")
    return split_chronologically_by_source(data, source_column="source_file", label_column=config.label_column)


def _load_experiment_and_data(
    experiment_config_path: str | Path, dataset_config_dir: str | Path
) -> tuple[dict[str, Any], DatasetConfig, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    experiment = load_yaml(experiment_config_path)

    strategy = experiment.get("preprocessing", {}).get("strategy", _SUPPORTED_PREPROCESSING_STRATEGY)
    if strategy != _SUPPORTED_PREPROCESSING_STRATEGY:
        raise NotImplementedError(f"Unknown preprocessing strategy: {strategy}")

    dataset_config = load_dataset_config(Path(dataset_config_dir) / f"{experiment['dataset']}.yaml")

    train_df = pd.read_csv(Path(dataset_config.processed_path) / "train.csv")
    test_df = pd.read_csv(Path(dataset_config.processed_path) / "test.csv")

    x_train, y_train = split_features_target(train_df, dataset_config.label_column)
    x_test, y_test = split_features_target(test_df, dataset_config.label_column)

    return experiment, dataset_config, x_train, y_train, x_test, y_test


def _build_pipeline(experiment: dict[str, Any], dataset_config: DatasetConfig, x_train: pd.DataFrame) -> Pipeline:
    random_state = experiment.get("random_state", 42)
    preprocessing = build_preprocessing_pipeline(x_train, categorical_columns=dataset_config.categorical_columns)
    model = build_model(experiment["model"], random_state=random_state, **experiment.get("model_params", {}))
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])


def run_experiment(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    metrics_dir: str | Path = "outputs/metrics",
) -> dict[str, float]:
    """Train a model per an experiment config and save the fitted pipeline + metrics."""
    experiment, dataset_config, x_train, y_train, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )

    if experiment["model"] in _MODELS_REQUIRING_RESAMPLING:
        x_train, y_train = balance_classes(x_train, y_train, random_state=experiment.get("random_state", 42))

    pipeline = _build_pipeline(experiment, dataset_config, x_train)
    train_model(pipeline, x_train, y_train)

    report = evaluate_classifier(pipeline, x_test, y_test)
    metrics = {name: _METRIC_EXTRACTORS[name](report) for name in experiment.get("metrics", [])}

    experiment_name = experiment["name"]
    model_path = ensure_directory(models_dir) / f"{experiment_name}.joblib"
    metrics_path = ensure_directory(metrics_dir) / f"{experiment_name}.json"
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def tune_hyperparameters(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    n_iter: int = 30,
    cv: int = 3,
    subsample_size: int = 200_000,
    n_jobs: int = 2,
) -> dict[str, Any]:
    """Search for good hyperparameters on the training split and write them into the
    experiment config's model_params, for run_experiment() to pick up on the next run.

    Datasets larger than subsample_size are stratify-subsampled for search tractability;
    the final model is still trained on the full training set by run_experiment().
    """
    experiment, dataset_config, x_train, y_train, _, _ = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )
    random_state = experiment.get("random_state", 42)

    if len(x_train) > subsample_size:
        x_search, y_search = stratified_subsample(
            x_train, y_train, subsample_size, min_per_class=cv, random_state=random_state
        )
    else:
        x_search, y_search = x_train, y_train

    if experiment["model"] in _MODELS_REQUIRING_RESAMPLING:
        x_search, y_search = balance_classes(x_search, y_search, random_state=random_state)

    pipeline = _build_pipeline(experiment, dataset_config, x_search)
    n_splits = min(cv, y_search.value_counts().min())

    # n_jobs is deliberately modest (not -1): spawning one worker process per core for
    # large subsamples has triggered Windows resource-exhaustion errors (WinError 1450).
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DISTRIBUTIONS[experiment["model"]],
        n_iter=n_iter,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(x_search, y_search)

    tuned_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}

    experiment["model_params"] = {**experiment.get("model_params", {}), **tuned_params}
    dump_yaml(experiment, experiment_config_path)

    return tuned_params


def generate_confusion_matrix(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    plots_dir: str | Path = "outputs/plots",
    normalize: bool = False,
) -> Path:
    """Plot a confusion matrix for an already-trained experiment's test predictions."""
    experiment, _, _, _, x_test, y_test = _load_experiment_and_data(experiment_config_path, dataset_config_dir)

    experiment_name = experiment["name"]
    model = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    predictions = model.predict(x_test)

    output_path = Path(plots_dir) / f"{experiment_name}_confusion_matrix.png"
    return plot_confusion_matrix(y_test, predictions, output_path, normalize=normalize)


def generate_label_distribution(
    dataset_config_path: str | Path,
    *,
    plots_dir: str | Path = "outputs/plots",
) -> Path:
    """Plot the combined train+test label distribution for a processed dataset."""
    dataset_config = load_dataset_config(dataset_config_path)

    train_df = pd.read_csv(Path(dataset_config.processed_path) / "train.csv")
    test_df = pd.read_csv(Path(dataset_config.processed_path) / "test.csv")
    labels = pd.concat([train_df[dataset_config.label_column], test_df[dataset_config.label_column]])

    output_path = Path(plots_dir) / f"{dataset_config.name}_label_distribution.png"
    return plot_label_distribution(labels, output_path, title=f"{dataset_config.name} label distribution")


# (target_class, confused_with_class) per dataset. For DAPT2020 (multiclass), informed by the
# confusion-matrix analysis: target_class is a class substantially confused with confused_with_class
# across every model trained, not just one - e.g. DAPT2020's Reconnaissance is misclassified as
# Benign by every model (10-75% of the time depending on model). Ties local explanations back to
# a concrete, already-established error pattern instead of an arbitrary/random instance choice. For
# the two binary datasets, Attack misclassified as Benign (a false negative - an undetected attack)
# is the single most security-critical error an IDS can make, so it's the natural class of interest
# regardless of per-model confusion specifics.
_CLASS_OF_INTEREST: dict[str, tuple[str, str]] = {
    "unsw_nb15": ("Attack", "Benign"),
    "cic_ids2017": ("Attack", "Benign"),
    "dapt2020": ("Reconnaissance", "Benign"),
}

# LIME's default discretize_continuous=True crashes on CIC-IDS2017's heavily-skewed columns
# (scipy truncnorm "scale must be positive" - confirmed reproducible, see explainability.py);
# verified safe for UNSW-NB15 specifically (30/30 test instances, no error) before enabling it
# there. DAPT2020 doesn't use this setting at all (it falls back to LIME's own single-panel
# rendering, since the fixed two-class layout doesn't generalize past two classes).
_SAFE_TO_DISCRETIZE_LIME = {"unsw_nb15"}


def generate_explanations(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    plots_dir: str | Path = "outputs/plots",
    sample_size: int = 200,
    pdp_top_n: int = 5,
) -> dict[str, Path]:
    """Generate SHAP, LIME, permutation importance, and partial dependence explanations for
    an already-trained experiment, saving plots under plots_dir/<experiment_name>/.
    """
    experiment, dataset_config, x_train, _, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )
    random_state = experiment.get("random_state", 42)
    experiment_name = experiment["name"]
    model_name = experiment["model"]

    if dataset_config.name not in _CLASS_OF_INTEREST:
        raise NotImplementedError(f"No class-of-interest mapping configured for dataset: {dataset_config.name}")
    target_class, confused_with_class = _CLASS_OF_INTEREST[dataset_config.name]

    pipeline = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    output_dir = ensure_directory(Path(plots_dir) / experiment_name)
    class_names = list(pipeline.named_steps["model"].classes_)

    if len(x_test) > sample_size:
        sample_x, _, sample_y, _ = train_test_split(
            x_test, y_test, train_size=sample_size, stratify=y_test, random_state=random_state
        )
    else:
        sample_x, sample_y = x_test, y_test
    background_x = x_train.sample(n=min(100, len(x_train)), random_state=random_state)

    paths: dict[str, Path] = {}

    shap_values, base_values, x_transformed, feature_names = compute_shap_values(
        pipeline, model_name, sample_x, background_x, categorical_columns=dataset_config.categorical_columns
    )
    paths["shap_bar"] = plot_shap_bar(
        shap_values, x_transformed, feature_names, class_names, output_dir / "shap_bar.png"
    )
    if target_class in class_names:
        target_index = class_names.index(target_class)
        paths["shap_beeswarm"] = plot_shap_beeswarm(
            shap_values, x_transformed, feature_names, target_index, output_dir / "shap_beeswarm.png"
        )

    pfi_path, top_features = compute_permutation_importance(
        pipeline, sample_x, sample_y, output_dir / "permutation_importance.png", random_state=random_state
    )
    paths["permutation_importance"] = pfi_path

    if target_class in class_names:
        # sklearn's PDP computes numeric deciles for its gridlines, which breaks outright
        # on categorical (string) columns; those would need a different plot style we
        # haven't built, so just restrict to numeric features here.
        numeric_top_features = [
            feature for feature in top_features if feature not in dataset_config.categorical_columns
        ]
        paths["partial_dependence"] = plot_partial_dependence(
            pipeline, sample_x, numeric_top_features[:pdp_top_n], target_class, output_dir / "partial_dependence.png"
        )

    instances = select_curated_instances(
        pipeline, x_test, y_test, target_class, confused_with_class, random_state=random_state
    )
    if instances:
        instance_kinds = list(instances.keys())
        instance_indices = list(instances.values())
        instances_x = x_test.loc[instance_indices]
        instance_shap_values, instance_base_values, instances_transformed, _ = compute_shap_values(
            pipeline, model_name, instances_x, background_x, categorical_columns=dataset_config.categorical_columns
        )
        predicted_classes = pipeline.predict(instances_x)

        for row_index, (kind, predicted_class) in enumerate(zip(instance_kinds, predicted_classes, strict=True)):
            class_index = class_names.index(predicted_class)
            paths[f"shap_waterfall_{kind}"] = plot_shap_waterfall(
                instance_shap_values,
                instance_base_values,
                instances_transformed[row_index],
                feature_names,
                row_index,
                class_index,
                output_dir / f"shap_waterfall_{kind}.png",
            )
            paths[f"lime_{kind}"] = explain_instance_with_lime(
                pipeline,
                x_train,
                x_test.loc[instance_indices[row_index]],
                class_names,
                output_dir / f"lime_{kind}.png",
                categorical_columns=dataset_config.categorical_columns,
                random_state=random_state,
                discretize_continuous=dataset_config.name in _SAFE_TO_DISCRETIZE_LIME,
            )

    return paths
