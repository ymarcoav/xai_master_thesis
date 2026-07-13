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
from xai_thesis.models import PARAM_DISTRIBUTIONS, build_model, train_model
from xai_thesis.preprocessing import (
    build_preprocessing_pipeline,
    split_by_official_split_column,
    split_chronologically_by_source,
    split_features_target,
)
from xai_thesis.utils import dump_yaml, ensure_directory, load_yaml

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
    preprocessing = build_preprocessing_pipeline(
        x_train, categorical_columns=dataset_config.categorical_columns, random_state=random_state
    )
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
    n_iter: int = 10,
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
        x_search, _, y_search, _ = train_test_split(
            x_train, y_train, train_size=subsample_size, stratify=y_train, random_state=random_state
        )
    else:
        x_search, y_search = x_train, y_train

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
    normalize: bool = True,
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
