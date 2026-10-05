"""End-to-end pipeline: preprocessing, training, evaluation, explanation and agreement analysis.

Each public function is driven by a script in scripts/ and reads its settings from the YAML
configs in configs/.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from xai_thesis.agreement import (
    AGREEMENT_METRIC_TITLES,
    AGREEMENT_METRICS,
    agreement_over_k,
    plot_agreement_heatmap,
    summarize_agreement,
)
from xai_thesis.datasets import DATASET_REGISTRY
from xai_thesis.datasets.base import DatasetConfig, load_dataset_config
from xai_thesis.evaluation import (
    confusion_matrix_frame,
    evaluate_classifier,
    plot_confusion_matrix,
    plot_feature_correlation,
    plot_label_distribution,
    plot_missing_values,
    plot_roc_curves,
    plot_train_test_split_distribution,
)
from xai_thesis.explainability import (
    _build_lime_explainer,
    compute_permutation_importance,
    compute_shap_values,
    explain_instance_with_lime,
    plot_shap_bar,
    plot_shap_waterfall,
    select_curated_instances,
)
from xai_thesis.models import PARAM_DISTRIBUTIONS, build_model, train_model
from xai_thesis.preprocessing import (
    build_preprocessing_pipeline,
    split_features_target,
    stratified_random_split,
    stratified_subsample,
)
from xai_thesis.screening import check_screen, screen_features
from xai_thesis.utils import dump_yaml, ensure_directory, load_yaml

logger = logging.getLogger(__name__)

# XGBClassifier has no class_weight parameter, so its balanced class weights are passed to fit()
# as per-sample weights (compute_sample_weight("balanced", y): the same n / (k * count_c) weights
# that the other models apply internally).
_MODELS_REQUIRING_SAMPLE_WEIGHT = {"xgboost"}

_METRIC_EXTRACTORS = {
    "accuracy": lambda report: report["accuracy"],
    "precision_macro": lambda report: report["macro avg"]["precision"],
    "recall_macro": lambda report: report["macro avg"]["recall"],
    "f1_macro": lambda report: report["macro avg"]["f1-score"],
    # One-vs-rest, macro-averaged (see evaluate_classifier()).
    "roc_auc_macro": lambda report: report["roc_auc_macro"],
    "pr_auc_macro": lambda report: report["pr_auc_macro"],
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
    # Stratified random 70/30 split with a fixed seed. "source_file", the loading tag every
    # adapter adds, is not a feature and is dropped here.
    return stratified_random_split(data, config.label_column, drop_columns=("source_file",))


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


# Model params used only during the hyperparameter search and never written to the config. The
# search scores with predict() alone, so the SVM's costly probability calibration is skipped.
_SEARCH_TIME_MODEL_PARAM_OVERRIDES: dict[str, dict[str, Any]] = {
    "svm": {"probability": False},
}


def _build_pipeline(experiment: dict[str, Any], dataset_config: DatasetConfig, x_train: pd.DataFrame) -> Pipeline:
    random_state = experiment.get("random_state", 42)
    preprocessing = build_preprocessing_pipeline(x_train, categorical_columns=dataset_config.categorical_columns)
    model = build_model(experiment["model"], random_state=random_state, **experiment.get("model_params", {}))
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])


# The SVM's final fit is intractable on the full training partitions (well over a million rows),
# so it is fitted on a stratified subsample given as (target_size, min_per_class): each class
# contributes its proportional share of target_size or min_per_class rows, whichever is larger,
# up to the number available. This keeps the rare attack classes usable. Both datasets are
# treated alike.
_FINAL_FIT_SUBSAMPLE: dict[tuple[str, str], tuple[int, int]] = {
    ("svm", "unsw_nb15"): (200_000, 500),
    ("svm", "cic_ids2017"): (200_000, 500),
}


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
    random_state = experiment.get("random_state", 42)

    subsample = _FINAL_FIT_SUBSAMPLE.get((experiment["model"], dataset_config.name))
    if subsample is not None:
        target_size, min_per_class = subsample
        x_train, y_train = stratified_subsample(
            x_train, y_train, target_size, min_per_class=min_per_class, random_state=random_state
        )

    pipeline = _build_pipeline(experiment, dataset_config, x_train)
    if experiment["model"] in _MODELS_REQUIRING_SAMPLE_WEIGHT:
        train_model(pipeline, x_train, y_train, model__sample_weight=compute_sample_weight("balanced", y_train))
    else:
        # random_forest and svm use class_weight="balanced"; the MLP computes the same weights
        # inside fit().
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

    RandomizedSearchCV samples n_iter configurations (the whole grid if it is smaller) and scores
    them by macro-F1 with stratified cv-fold cross-validation. Training sets larger than
    subsample_size are first reduced to a class-proportional subsample.
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

    # No per-model resampling here anymore: every model reweights its objective (class_weight,
    # sample_weight, or TorchMLPClassifier's per-class CrossEntropyLoss weight), so each
    # RandomizedSearchCV fold's fit sees the real fold rows and computes its own weights from
    # them - nothing to leak across the split.
    search_overrides = _SEARCH_TIME_MODEL_PARAM_OVERRIDES.get(experiment["model"], {})
    search_experiment = {**experiment, "model_params": {**experiment.get("model_params", {}), **search_overrides}}
    pipeline = _build_pipeline(search_experiment, dataset_config, x_search)
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
    # Per-sample fit params are sliced per CV fold by RandomizedSearchCV, so the weights can be
    # computed once over the whole search set.
    fit_params = (
        {"model__sample_weight": compute_sample_weight("balanced", y_search)}
        if experiment["model"] in _MODELS_REQUIRING_SAMPLE_WEIGHT
        else {}
    )
    search.fit(x_search, y_search, **fit_params)

    tuned_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}

    experiment["model_params"] = {**experiment.get("model_params", {}), **tuned_params}
    dump_yaml(experiment, experiment_config_path)

    return tuned_params


# Layout of outputs/plots/:
#   <dataset>/data/                                label distribution, missing values, split, correlation
#   <dataset>/agreement/                           local_agreement_{feature,rank}.png (SHAP vs LIME)
#   <dataset>/<model>_baseline/performance/        confusion_matrix.png, roc_curves.png
#   <dataset>/<model>_baseline/global/             shap_bar, permutation_importance
#   <dataset>/<model>_baseline/interview/<class>/  shap_waterfall_*, lime_* (interview stimuli)


def _dataset_plot_dir(plots_dir: str | Path, dataset_name: str, section: str) -> Path:
    """One purpose-named folder of dataset-level plots ("data" or "agreement"); see the layout above."""
    return Path(plots_dir) / dataset_name / section


def _experiment_plot_dir(plots_dir: str | Path, dataset_name: str, experiment_name: str, section: str) -> Path:
    """One purpose-named folder ("performance", "global" or "interview") of a trained experiment,
    plots_dir/<dataset>/<model>_baseline/<section>; see the layout above. Experiment names follow
    "<dataset>_<model>_baseline".
    """
    model_folder = experiment_name.removeprefix(f"{dataset_name}_")
    return Path(plots_dir) / dataset_name / model_folder / section


def generate_confusion_matrix(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    plots_dir: str | Path = "outputs/plots",
    normalize: bool = False,
) -> Path:
    """Plot a confusion matrix for an already-trained experiment's test predictions, and write
    its counts next to the PNG as confusion_matrix.csv (rows true class, columns predicted), so
    the counts quoted in the thesis can be read back without recomputing predictions."""
    experiment, dataset_config, _, _, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )

    experiment_name = experiment["name"]
    model = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    predictions = model.predict(x_test)

    output_dir = ensure_directory(_experiment_plot_dir(plots_dir, dataset_config.name, experiment_name, "performance"))
    confusion_matrix_frame(y_test, predictions).to_csv(output_dir / "confusion_matrix.csv")
    return plot_confusion_matrix(y_test, predictions, output_dir / "confusion_matrix.png", normalize=normalize)


def generate_roc_curves(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    plots_dir: str | Path = "outputs/plots",
) -> Path:
    """Plot the aggregate one-vs-rest ROC curves (macro- and micro-average + chance diagonal)
    for an already-trained experiment's test predictions, alongside its confusion matrix."""
    experiment, dataset_config, _, _, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )

    experiment_name = experiment["name"]
    model = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    probabilities = model.predict_proba(x_test)

    output_dir = _experiment_plot_dir(plots_dir, dataset_config.name, experiment_name, "performance")
    return plot_roc_curves(y_test, probabilities, list(model.classes_), output_dir / "roc_curves.png")


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

    output_path = _dataset_plot_dir(plots_dir, dataset_config.name, "data") / "label_distribution.png"
    return plot_label_distribution(labels, output_path)


def generate_raw_label_distribution(
    dataset_config_path: str | Path,
    *,
    plots_dir: str | Path = "outputs/plots",
    metrics_dir: str | Path = "outputs/metrics",
) -> dict[str, Path]:
    """Plot the class distribution over every raw record, after label normalization (and, for
    CIC-IDS2017, consolidation), before any row is removed.

    Unlike generate_label_distribution (processed train+test, so after corrupted-row removal and
    deduplication), this reads the raw files. Also writes the counts as
    metrics_dir/dataset_inspection/<dataset>/label_distribution_raw.csv (class, count, share).
    """
    config = load_dataset_config(dataset_config_path)
    dataset = DATASET_REGISTRY[config.name](config)
    labels = dataset.raw_labels(dataset.load_raw())

    counts = labels.value_counts()
    csv_path = ensure_directory(Path(metrics_dir) / "dataset_inspection" / config.name) / "label_distribution_raw.csv"
    pd.DataFrame({"class": counts.index, "count": counts.to_numpy(), "share": counts.to_numpy() / len(labels)}).to_csv(
        csv_path, index=False
    )
    plot_path = plot_label_distribution(
        labels, _dataset_plot_dir(plots_dir, config.name, "data") / "label_distribution_raw.png"
    )
    return {"plot": plot_path, "csv": csv_path}


def generate_feature_screen(
    dataset_config_path: str | Path,
    *,
    metrics_dir: str | Path = "outputs/metrics",
) -> tuple[Path, list[str]]:
    """Rerun the constant / near-duplicate screen and check it against screen_drop_columns.

    The dataset is cleaned exactly as for training except that screen_drop_columns is kept, so
    the screen sees the columns it is meant to remove. Every numeric feature is screened
    (categorical columns and the label are left out). Writes
    metrics_dir/dataset_inspection/<dataset>/feature_screen.csv and returns its path with the
    list of inconsistencies (see screening.check_screen) - empty when the config matches.
    """
    config = load_dataset_config(dataset_config_path)
    data = DATASET_REGISTRY[config.name](replace(config, screen_drop_columns=())).build()
    features = data.drop(columns=[config.label_column, "source_file", *config.categorical_columns], errors="ignore")

    screen = screen_features(features)
    screen.insert(0, "dataset", config.name)
    screen.insert(1, "n_rows", len(features))
    screen["in_screen_drop_columns"] = [
        [name for name in (row.column, row.other_column) if name in config.screen_drop_columns]
        for row in screen.itertuples(index=False)
    ]
    screen["in_screen_drop_columns"] = screen["in_screen_drop_columns"].map("; ".join)
    csv_path = ensure_directory(Path(metrics_dir) / "dataset_inspection" / config.name) / "feature_screen.csv"
    screen.to_csv(csv_path, index=False, float_format="%.6g")
    return csv_path, check_screen(screen, config.screen_drop_columns)


def generate_missing_value_summary(
    dataset_config_path: str | Path,
    *,
    plots_dir: str | Path = "outputs/plots",
) -> Path:
    """Plot per-column missing-value percentages for a processed dataset's combined train+test
    data. Missing values are imputed later, inside the training pipeline.
    """
    dataset_config = load_dataset_config(dataset_config_path)

    train_df = pd.read_csv(Path(dataset_config.processed_path) / "train.csv")
    test_df = pd.read_csv(Path(dataset_config.processed_path) / "test.csv")
    combined = pd.concat([train_df, test_df], ignore_index=True)

    output_path = _dataset_plot_dir(plots_dir, dataset_config.name, "data") / "missing_values.png"
    return plot_missing_values(combined, output_path, title=f"{dataset_config.name} missing values")


def generate_train_test_split_comparison(
    dataset_config_path: str | Path,
    *,
    plots_dir: str | Path = "outputs/plots",
) -> Path:
    """Plot per-class train vs test counts for a processed dataset's split."""
    dataset_config = load_dataset_config(dataset_config_path)

    train_df = pd.read_csv(Path(dataset_config.processed_path) / "train.csv")
    test_df = pd.read_csv(Path(dataset_config.processed_path) / "test.csv")

    output_path = _dataset_plot_dir(plots_dir, dataset_config.name, "data") / "train_test_split.png"
    return plot_train_test_split_distribution(
        train_df[dataset_config.label_column],
        test_df[dataset_config.label_column],
        output_path,
        title=f"{dataset_config.name} train/test split",
    )


def generate_feature_correlation(
    dataset_config_path: str | Path,
    *,
    plots_dir: str | Path = "outputs/plots",
    sample_size: int = 200_000,
) -> Path:
    """Plot a correlation heatmap over numeric feature columns for a processed dataset's
    training split.

    At most sample_size randomly drawn rows are used, which keeps the computation cheap on the
    large training partitions.
    """
    dataset_config = load_dataset_config(dataset_config_path)

    train_df = pd.read_csv(Path(dataset_config.processed_path) / "train.csv")
    features = train_df.drop(columns=[dataset_config.label_column])
    if len(features) > sample_size:
        features = features.sample(n=sample_size, random_state=42)

    output_path = _dataset_plot_dir(plots_dir, dataset_config.name, "data") / "feature_correlation.png"
    return plot_feature_correlation(features, output_path, title=f"{dataset_config.name} feature correlation")


# Global explanations (SHAP and permutation importance) are computed on a class-balanced sample
# of the test partition, so that every class is represented in the global rankings: a
# prevalence-proportional sample would be 85-96% Benign. A class with fewer test rows than its
# even share contributes all of them (CIC-IDS2017 Infiltration: 11).
#
# The SHAP background (reference distribution) is a class-balanced sample of the training
# partition. Only the permutation explainer (svm, neural_network) uses it; its cost grows with
# the background size, so the background is kept small.
_SHAP_BACKGROUND_SIZE = 50
_GLOBAL_EXPLANATION_SAMPLE_SIZE = 500

# (dataset, model) pairs whose global SHAP generate_explanations skips unless the experiment
# config sets `skip_global_shap: false`; permutation importance is still computed. For these,
# compute global SHAP once with scripts/explain.py --shap-chunk-size/--shap-n-jobs (checkpointed,
# parallel); the cached result is then reused. Any other experiment computes global SHAP unless
# its config sets `skip_global_shap: true`.
_GLOBAL_SHAP_SKIPPED_BY_DEFAULT: dict[tuple[str, str], str] = {
    ("cic_ids2017", "svm"): "permutation SHAP ~440 s/instance, ~53 h for the 437-row global sample",
}


def _skip_global_shap(experiment: dict[str, Any], dataset_name: str) -> bool:
    """The experiment config's skip_global_shap flag, defaulting per _GLOBAL_SHAP_SKIPPED_BY_DEFAULT."""
    default = (dataset_name, experiment["model"]) in _GLOBAL_SHAP_SKIPPED_BY_DEFAULT
    return bool(experiment.get("skip_global_shap", default))


def _global_importance_path(metrics_dir: str | Path, experiment_name: str, method: str) -> Path:
    """Cache of one experiment's global importance vector; method is "shap" or "pfi". Separate
    files let a rerun that skips global SHAP keep the existing SHAP cache.
    """
    return ensure_directory(Path(metrics_dir) / "global_importance") / f"{experiment_name}__{method}.npz"


def _save_global_importance(
    cache_path: Path,
    importance: np.ndarray,
    feature_names: Sequence[str],
    sample_y: pd.Series,
) -> Path:
    """Write a global importance vector (raw feature order) with the exact sample it came from."""
    class_counts = sample_y.value_counts()
    np.savez(
        cache_path,
        importance=np.asarray(importance, dtype=float),
        feature_names=np.array(list(feature_names)),
        instance_index=sample_y.index.to_numpy(),
        class_names=np.array([str(label) for label in class_counts.index]),
        class_counts=class_counts.to_numpy(),
    )
    return cache_path


def _log_sample(label: str, sample_y: pd.Series) -> None:
    counts = sample_y.value_counts().sort_index()
    logger.info(
        "%s: n=%d, per class: %s", label, len(sample_y), ", ".join(f"{name}={n}" for name, n in counts.items())
    )


def _class_balanced_sample(
    features: pd.DataFrame, target: pd.Series, total: int, *, random_state: int
) -> pd.DataFrame:
    """Sample ~total rows from features, spread evenly across the classes in target (a class
    with fewer rows than its even share contributes all of them).

    Used for the global and local explanation samples and for the SHAP background. A balanced
    background keeps the permutation explainer's base value close to that of the tree models,
    whose class-weighted training already gives them a near-balanced prior; a uniform random
    background would be dominated by Benign flows.
    """
    groups = [
        features.loc[index_block].sample(n=min(total // target.nunique() or 1, len(index_block)), random_state=random_state)
        for index_block in (target.index[target == label] for label in target.unique())
    ]
    return pd.concat(groups)


def _shap_chunk(
    pipeline: Pipeline,
    model_name: str,
    chunk_x: pd.DataFrame,
    background_x: pd.DataFrame,
    categorical_columns: Sequence[str],
    seed: int,
    chunk_path: Path,
) -> Path:
    """Compute one chunk's SHAP values and save them to chunk_path (a joblib worker's unit of work)."""
    shap_values, base_values, _, feature_names = compute_shap_values(
        pipeline, model_name, chunk_x, background_x, categorical_columns=categorical_columns, random_state=seed
    )
    # Written under a temporary name and renamed, so an interrupted write never looks like a
    # finished chunk to the resume check.
    partial_path = chunk_path.with_name(f"{chunk_path.stem}.partial.npz")
    np.savez(
        partial_path,
        shap_values=shap_values,
        base_values=base_values,
        feature_names=np.array(feature_names),
        instance_index=chunk_x.index.to_numpy(),
    )
    partial_path.replace(chunk_path)
    return chunk_path


def _compute_shap_in_chunks(
    pipeline: Pipeline,
    model_name: str,
    sample_x: pd.DataFrame,
    background_x: pd.DataFrame,
    *,
    categorical_columns: Sequence[str],
    random_state: int,
    chunk_size: int,
    n_jobs: int,
    chunk_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """compute_shap_values over sample_x in fixed-size chunks, checkpointed and run in parallel.

    Intended for the permutation explainer, whose cost is per instance (several minutes per
    instance for the CIC-IDS2017 SVM). Each chunk is saved to chunk_dir when it finishes and a
    rerun skips chunks already on disk, so an interrupted run resumes; chunks run n_jobs at a time.

    Chunk c is seeded random_state + c. The permutation explainer draws from one random stream
    per call, so chunked values are not bit-identical to a single call, although the estimator
    (permutation count, background) is the same and the result is reproducible for a given
    chunk_size.
    """
    chunk_dir = ensure_directory(chunk_dir)
    starts = list(range(0, len(sample_x), chunk_size))
    chunk_paths = [chunk_dir / f"chunk_{number:03d}.npz" for number in range(len(starts))]
    pending = [(number, start) for number, start in enumerate(starts) if not chunk_paths[number].exists()]
    # max_nbytes=None: joblib would otherwise pass large arguments as read-only memory maps,
    # which libsvm (the SVM's predict_proba) rejects.
    joblib.Parallel(n_jobs=n_jobs, verbose=10, max_nbytes=None)(
        joblib.delayed(_shap_chunk)(
            pipeline,
            model_name,
            sample_x.iloc[start : start + chunk_size],
            background_x,
            categorical_columns,
            random_state + number,
            chunk_paths[number],
        )
        for number, start in pending
    )

    chunks = [np.load(path, allow_pickle=False) for path in chunk_paths]
    instance_index = np.concatenate([chunk["instance_index"] for chunk in chunks])
    if not np.array_equal(instance_index, sample_x.index.to_numpy()):
        raise ValueError(f"chunks in {chunk_dir} do not cover this sample in order - clear it and rerun")
    feature_names = [str(name) for name in chunks[0]["feature_names"]]
    return (
        np.concatenate([chunk["shap_values"] for chunk in chunks]),
        np.concatenate([chunk["base_values"] for chunk in chunks]),
        # The same display_data compute_shap_values returns: the original rows, raw columns.
        sample_x[feature_names].to_numpy(dtype=object),
        feature_names,
    )


def generate_explanations(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    metrics_dir: str | Path = "outputs/metrics",
    plots_dir: str | Path = "outputs/plots",
    sample_size: int = _GLOBAL_EXPLANATION_SAMPLE_SIZE,
    shap_chunk_size: int | None = None,
    shap_n_jobs: int = 1,
) -> dict[str, Path]:
    """Generate the global SHAP bar and permutation importance plots for an already-trained
    experiment, saving them under plots_dir/<dataset>/<model>_baseline/global/.

    Both explain the same sample_size-row class-balanced sample of the test partition
    (_class_balanced_sample). Global SHAP importance is the mean |SHAP| over instances and
    classes; permutation importance is the drop in macro-F1 over 10 repetitions. Both vectors
    are cached under metrics_dir/global_importance/.

    shap_chunk_size splits the global SHAP computation into checkpointed chunks run shap_n_jobs
    at a time (see _compute_shap_in_chunks); None computes it in one call.
    """
    experiment, dataset_config, x_train, y_train, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )
    random_state = experiment.get("random_state", 42)
    experiment_name = experiment["name"]
    model_name = experiment["model"]

    pipeline = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    output_dir = ensure_directory(_experiment_plot_dir(plots_dir, dataset_config.name, experiment_name, "global"))
    class_names = list(pipeline.named_steps["model"].classes_)

    # Local SHAP/LIME plots are produced by generate_interview_stimuli().

    sample_x = _class_balanced_sample(x_test, y_test, sample_size, random_state=random_state)
    sample_y = y_test.loc[sample_x.index]
    background_x = _class_balanced_sample(
        x_train, y_train, _SHAP_BACKGROUND_SIZE, random_state=random_state
    )

    paths: dict[str, Path] = {}
    _log_sample(f"{experiment_name} global sample (SHAP and PFI)", sample_y)

    if _skip_global_shap(experiment, dataset_config.name):
        logger.warning(
            "%s: global SHAP skipped (skip_global_shap; default reason: %s); computing PFI only",
            experiment_name,
            _GLOBAL_SHAP_SKIPPED_BY_DEFAULT.get((dataset_config.name, model_name), "set in the experiment config"),
        )
    else:
        if shap_chunk_size is None:
            shap_values, base_values, x_transformed, feature_names = compute_shap_values(
                pipeline,
                model_name,
                sample_x,
                background_x,
                categorical_columns=dataset_config.categorical_columns,
                random_state=random_state,
            )
        else:
            shap_values, base_values, x_transformed, feature_names = _compute_shap_in_chunks(
                pipeline,
                model_name,
                sample_x,
                background_x,
                categorical_columns=dataset_config.categorical_columns,
                random_state=random_state,
                chunk_size=shap_chunk_size,
                n_jobs=shap_n_jobs,
                chunk_dir=Path(metrics_dir)
                / "global_importance"
                / "chunks"
                / f"{experiment_name}__balanced{sample_size}__chunk{shap_chunk_size}",
            )
        paths["shap_bar"] = plot_shap_bar(
            shap_values, x_transformed, feature_names, class_names, output_dir / "shap_bar.png"
        )
        # Global SHAP importance: mean |SHAP| over instances and classes, in raw column order.
        # (plot_shap_bar stacks the per-class means, so its bars are n_classes times longer.)
        shap_order = [feature_names.index(name) for name in sample_x.columns]
        paths["shap_cache"] = _save_global_importance(
            _global_importance_path(metrics_dir, experiment_name, "shap"),
            np.abs(shap_values).mean(axis=(0, 2))[shap_order],
            list(sample_x.columns),
            sample_y,
        )

    # The sample is class-balanced, so permutation importance measures the macro-F1 drop under
    # an even class distribution, matching the global SHAP sample.
    pfi_path, _, pfi_result = compute_permutation_importance(
        pipeline, sample_x, sample_y, output_dir / "permutation_importance.png", random_state=random_state
    )
    paths["permutation_importance"] = pfi_path
    paths["pfi_cache"] = _save_global_importance(
        _global_importance_path(metrics_dir, experiment_name, "pfi"),
        pfi_result.importances_mean,
        list(sample_x.columns),
        sample_y,
    )

    return paths


def _generate_instance_explanations(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    background_x: pd.DataFrame,
    target_class: str,
    confused_with_class: str,
    class_names: list[str],
    model_name: str,
    categorical_columns: Sequence[str],
    output_dir: Path,
    random_state: int,
    *,
    predictions: pd.Series | None = None,
) -> dict[str, Path]:
    """SHAP waterfall and LIME plots for one correctly classified instance of target_class and
    one misclassified as confused_with_class. Used by generate_interview_stimuli(). Pass
    predictions to reuse an existing prediction of the whole test set.
    """
    paths: dict[str, Path] = {}
    instances = select_curated_instances(
        pipeline, x_test, y_test, target_class, confused_with_class,
        predictions=predictions, random_state=random_state,
    )
    if not instances:
        return paths

    instance_kinds = list(instances.keys())
    instance_indices = list(instances.values())
    instances_x = x_test.loc[instance_indices]
    instance_shap_values, instance_base_values, instances_transformed, feature_names = compute_shap_values(
        pipeline,
        model_name,
        instances_x,
        background_x,
        categorical_columns=categorical_columns,
        random_state=random_state,
    )
    predicted_classes = pipeline.predict(instances_x)

    # XGBoost's SHAP values are in log-odds; the other models' are probabilities (see
    # compute_shap_values).
    waterfall_space = "log-odds" if model_name == "xgboost" else "probability"

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
            value_label=waterfall_space,
        )
        # LIME explains the same class as the SHAP waterfall: the predicted class. Without
        # class_index, LIME would pick the most probable class, which for the SVM can differ
        # from its prediction (one-vs-one voting).
        paths[f"lime_{kind}"] = explain_instance_with_lime(
            pipeline,
            x_train,
            x_test.loc[instance_indices[row_index]],
            class_names,
            output_dir / f"lime_{kind}.png",
            categorical_columns=categorical_columns,
            random_state=random_state,
            class_index=class_index,
        )

    return paths


def _most_common_confusion(y_test: pd.Series, predictions: pd.Series, target_class: str) -> str | None:
    """The class that target_class is most often misclassified as, or None if it is never
    misclassified in the test set. Used by generate_interview_stimuli().
    """
    is_target = y_test == target_class
    wrong = predictions[is_target & (predictions != target_class)]
    if len(wrong) == 0:
        return None
    return wrong.value_counts().idxmax()


# Candidate classes for interview stimuli: attack types a security practitioner recognizes by
# name. Excluded: classes with very few records (UNSW-NB15 Worms, CIC-IDS2017 Infiltration),
# category names that are not common security terms (Generic, Analysis, Backdoor, Shellcode),
# umbrella labels that mix several techniques (Web Attack), and CIC-IDS2017 PortScan, which
# deduplication reduces to a small, unrepresentative sample once Destination Port is removed.
_INTERVIEW_STIMULUS_CLASSES: dict[str, list[str]] = {
    "unsw_nb15": ["Reconnaissance", "Exploits"],
    "cic_ids2017": ["DoS/DDoS", "Brute Force", "Bot"],
}


def generate_interview_stimuli(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    plots_dir: str | Path = "outputs/plots",
) -> dict[str, dict[str, Path]]:
    """Generate SHAP waterfall and LIME plots for the interview stimuli of a trained experiment:
    one correctly classified and one misclassified instance per class in
    _INTERVIEW_STIMULUS_CLASSES.

    Saves under plots_dir/<dataset>/<model>_baseline/interview/<class>/. The result maps each
    class to its plot paths ("shap_waterfall_correct", "lime_correct", ...), or to an empty dict
    if the model never misclassifies that class.
    """
    experiment, dataset_config, x_train, y_train, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )
    random_state = experiment.get("random_state", 42)
    experiment_name = experiment["name"]
    model_name = experiment["model"]

    stimulus_classes = _INTERVIEW_STIMULUS_CLASSES.get(dataset_config.name, [])
    if not stimulus_classes:
        raise NotImplementedError(f"No interview stimulus classes configured for dataset: {dataset_config.name}")

    pipeline = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    class_names = list(pipeline.named_steps["model"].classes_)
    predictions = pd.Series(pipeline.predict(x_test), index=x_test.index)
    background_x = _class_balanced_sample(
        x_train, y_train, _SHAP_BACKGROUND_SIZE, random_state=random_state
    )
    interview_dir = _experiment_plot_dir(plots_dir, dataset_config.name, experiment_name, "interview")

    results: dict[str, dict[str, Path]] = {}
    for target_class in stimulus_classes:
        if target_class not in class_names:
            continue
        confused_with_class = _most_common_confusion(y_test, predictions, target_class)
        if confused_with_class is None:
            results[target_class] = {}
            continue
        output_dir = ensure_directory(interview_dir / target_class.replace("/", "-"))
        results[target_class] = _generate_instance_explanations(
            pipeline,
            x_train,
            x_test,
            y_test,
            background_x,
            target_class,
            confused_with_class,
            class_names,
            model_name,
            dataset_config.categorical_columns,
            output_dir,
            random_state,
            predictions=predictions,
        )
    return results


# --- Local disagreement analysis: SHAP vs LIME (Krishna et al. 2022) ------------
# Globally, the only comparison is the number of features shared by the SHAP and permutation
# importance top 10s (explanation_report.global_overlap_table).

# Instances the agreement metrics are averaged over: a class-balanced sample of the test
# partition (60 on UNSW-NB15, 6 per class; 56 on CIC-IDS2017, 8 per class), included whether or
# not the model classifies them correctly. The draw depends only on the test partition and the
# seed, so all four models of a dataset are compared on the same instances.
_AGREEMENT_SAMPLE_SIZE = 60

# Top-k values reported; k = 10 is the headline value.
_AGREEMENT_K_VALUES = (1, 5, 10)

# Fixed row order for the heatmaps, so panels line up across datasets and metrics.
_AGREEMENT_MODEL_ORDER = ("random_forest", "xgboost", "neural_network", "svm")

# (dataset, model) pairs to leave out of the local agreement analysis, each with a reason that is
# printed in the heatmap footnote. Empty: every pair is computed.
_AGREEMENT_EXCLUDED: dict[tuple[str, str], str] = {}


def _partition_excluded(
    dataset_name: str,
    experiments: Sequence[tuple[str, Path]],
    models: Sequence[str] | None = None,
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Split (model, config) pairs into those to compute and a note per model left out.

    A model is left out if it is listed in _AGREEMENT_EXCLUDED or not in the `models` filter.
    """
    included: list[tuple[str, Path]] = []
    notes: list[str] = []
    for model, path in experiments:
        if (dataset_name, model) in _AGREEMENT_EXCLUDED:
            notes.append(f"{model} omitted: {_AGREEMENT_EXCLUDED[(dataset_name, model)]}")
        elif models is not None and model not in models:
            notes.append(f"{model} not requested")
        else:
            included.append((model, path))
    if not included:
        raise ValueError(f"no models left to compare for {dataset_name}: {notes}")
    return included, notes


def _align_attribution_columns(
    shap_feature_names: Sequence[str], lime_columns: Sequence[str]
) -> tuple[list[int], list[int], list[str]]:
    """Index maps putting SHAP and LIME attributions on one shared feature order.

    SHAP attributions are mapped back onto the raw columns (see compute_shap_values), and LIME
    works on the raw columns minus any zero-variance column it drops. For both datasets the two
    name sets are identical, so this is a safeguard.
    """
    shap_position = {name: index for index, name in enumerate(shap_feature_names)}
    lime_position = {name: index for index, name in enumerate(lime_columns)}
    shared = [name for name in shap_feature_names if name in lime_position]
    if not shared:
        raise ValueError("SHAP and LIME feature-name sets do not overlap at all")
    return [shap_position[name] for name in shared], [lime_position[name] for name in shared], shared


def compute_explanation_attributions(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    metrics_dir: str | Path = "outputs/metrics",
    n_instances: int = _AGREEMENT_SAMPLE_SIZE,
    overwrite: bool = False,
) -> Path:
    """Compute and cache the per-instance SHAP and LIME attribution matrices one experiment
    contributes to the agreement heatmaps.

    Writes metrics_dir/agreement/attributions/<experiment>.npz with two [n_instances, n_features]
    signed matrices on a shared feature order, plus the feature names, the test-set row index and
    the class each instance was explained for. The result is cached because the permutation
    explainer is slow for the SVM.

    Both methods explain the same class for a given instance: the class with the highest
    predicted probability (LIME's top_labels=1).
    """
    experiment, dataset_config, x_train, y_train, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )
    experiment_name = experiment["name"]
    cache_path = ensure_directory(Path(metrics_dir) / "agreement" / "attributions") / f"{experiment_name}.npz"
    if cache_path.exists() and not overwrite:
        return cache_path

    random_state = experiment.get("random_state", 42)
    model_name = experiment["model"]
    categorical_columns = dataset_config.categorical_columns

    pipeline = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    class_names = list(pipeline.named_steps["model"].classes_)
    sample_x = _class_balanced_sample(x_test, y_test, n_instances, random_state=random_state)
    _log_sample(f"{experiment_name} local sample (SHAP vs LIME)", y_test.loc[sample_x.index])
    background_x = _class_balanced_sample(x_train, y_train, _SHAP_BACKGROUND_SIZE, random_state=random_state)

    shap_values, _, _, shap_feature_names = compute_shap_values(
        pipeline,
        model_name,
        sample_x,
        background_x,
        categorical_columns=categorical_columns,
        random_state=random_state,
    )

    # One LIME explainer for all instances (building it scans x_train). num_features covers every
    # column, so the full weight vector is returned rather than LIME's default top 8.
    lime_explainer, lime_predict_fn, lime_encode_instance, lime_columns = _build_lime_explainer(
        pipeline,
        x_train,
        class_names,
        categorical_columns=categorical_columns,
        random_state=random_state,
    )
    lime_rows: list[np.ndarray] = []
    explained_class: list[int] = []
    for instance_index in sample_x.index:
        explanation = lime_explainer.explain_instance(
            lime_encode_instance(x_test.loc[instance_index]),
            lime_predict_fn,
            num_features=len(lime_columns),
            top_labels=1,
        )
        label = explanation.available_labels()[0]
        weights = np.zeros(len(lime_columns))
        for feature_index, weight in explanation.local_exp[label]:
            weights[feature_index] = weight
        lime_rows.append(weights)
        explained_class.append(label)

    shap_index, lime_index, shared_names = _align_attribution_columns(shap_feature_names, lime_columns)
    shap_matrix = np.stack(
        [shap_values[row, shap_index, explained_class[row]] for row in range(len(sample_x))]
    )
    lime_matrix = np.stack([row[lime_index] for row in lime_rows])

    np.savez(
        cache_path,
        shap=shap_matrix,
        lime=lime_matrix,
        feature_names=np.array(shared_names),
        instance_index=sample_x.index.to_numpy(),
        explained_class=np.array([class_names[label] for label in explained_class]),
    )
    return cache_path


def _dataset_experiment_paths(dataset_name: str, config_dir: str | Path) -> list[tuple[str, Path]]:
    """(model_name, config_path) for every experiment on one dataset, in _AGREEMENT_MODEL_ORDER."""
    found: dict[str, Path] = {}
    for config_path in sorted(Path(config_dir).glob("*.yaml")):
        experiment = load_yaml(config_path)
        if experiment.get("dataset") == dataset_name:
            found[experiment["model"]] = config_path
    ordered = [(model, found[model]) for model in _AGREEMENT_MODEL_ORDER if model in found]
    ordered += [(model, path) for model, path in sorted(found.items()) if model not in _AGREEMENT_MODEL_ORDER]
    if not ordered:
        raise ValueError(f"No experiment configs in {config_dir} for dataset {dataset_name!r}")
    return ordered


def generate_agreement_heatmaps(
    dataset_name: str,
    *,
    config_dir: str | Path = "configs/experiments",
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    metrics_dir: str | Path = "outputs/metrics",
    plots_dir: str | Path = "outputs/plots",
    k_values: Sequence[int] = _AGREEMENT_K_VALUES,
    n_instances: int = _AGREEMENT_SAMPLE_SIZE,
    models: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Feature- and rank-agreement heatmaps between SHAP and LIME for one dataset.

    Rows are the dataset's models, columns are top-k values, and each cell is the metric
    averaged over the shared class-balanced instance set. The CSV adds the standard error
    (sample SD with ddof=1, divided by sqrt(n)) and n. Ties in |attribution| are broken by
    feature name (see agreement.ranked_feature_indices). Raises if two models' cached instance
    sets differ.

    Returns {"feature": path, "rank": path, "csv": path}; the CSV has one row per (model, k):
    dataset, model, k, fa_mean, fa_se, ra_mean, ra_se, n.
    """
    experiments, excluded_notes = _partition_excluded(
        dataset_name, _dataset_experiment_paths(dataset_name, config_dir), models
    )
    k_values = list(k_values)

    # The instance draw is seeded from each experiment's random_state, and every model must be
    # scored on the same instances, so the seeds must agree.
    seeds = {model: load_yaml(path).get("random_state", 42) for model, path in experiments}
    if len(set(seeds.values())) > 1:
        raise ValueError(
            f"agreement heatmaps need one shared random_state across all {dataset_name} "
            f"experiments (the instance draw depends on it); got {seeds}"
        )

    means = {metric: np.zeros((len(experiments), len(k_values))) for metric in AGREEMENT_METRICS}
    sems = {metric: np.zeros((len(experiments), len(k_values))) for metric in AGREEMENT_METRICS}
    rows: list[dict[str, Any]] = []
    reference_instances: np.ndarray | None = None

    for row, (model_name, config_path) in enumerate(experiments):
        cache_path = compute_explanation_attributions(
            config_path,
            dataset_config_dir=dataset_config_dir,
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            n_instances=n_instances,
            overwrite=overwrite,
        )
        cached = np.load(cache_path, allow_pickle=False)
        shap_matrix, lime_matrix = cached["shap"], cached["lime"]
        feature_names = [str(name) for name in cached["feature_names"]]
        if reference_instances is None:
            reference_instances = cached["instance_index"]
        elif not np.array_equal(cached["instance_index"], reference_instances):
            raise ValueError(f"{dataset_name}/{model_name}: instance set differs from the other models'")
        for metric in AGREEMENT_METRICS:
            metric_means, metric_sems = agreement_over_k(shap_matrix, lime_matrix, k_values, metric, feature_names)
            means[metric][row, :] = metric_means
            sems[metric][row, :] = metric_sems
        rows.extend(
            {"dataset": dataset_name, "model": model_name, **summary}
            for summary in summarize_agreement(shap_matrix, lime_matrix, k_values, feature_names)
        )

    output_dir = ensure_directory(_dataset_plot_dir(plots_dir, dataset_name, "agreement"))
    paths = {
        metric: plot_agreement_heatmap(
            means[metric],
            sems[metric],
            [model for model, _ in experiments],
            k_values,
            output_dir / f"local_agreement_{metric}.png",
            metric=metric,
            title=f"{AGREEMENT_METRIC_TITLES[metric]} - SHAP vs LIME, {dataset_name}",
            footnote="; ".join(excluded_notes),
        )
        for metric in AGREEMENT_METRICS
    }

    csv_path = ensure_directory(Path(metrics_dir) / "agreement") / f"{dataset_name}_local_agreement.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths["csv"] = csv_path
    return paths


# --- Sample composition ------------------------------------------------------------


def sample_composition(
    dataset_name: str,
    *,
    config_dir: str | Path = "configs/experiments",
    dataset_config_dir: str | Path = "configs/datasets",
    metrics_dir: str | Path | None = "outputs/metrics",
    local_size: int = _AGREEMENT_SAMPLE_SIZE,
    global_size: int = _GLOBAL_EXPLANATION_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Per-class row counts of the local (SHAP vs LIME) and global (SHAP/PFI) samples.

    Redraws both samples exactly as the pipeline does (_class_balanced_sample on the test split,
    with the experiments' shared random_state) and, when metrics_dir is given, checks every
    cached result against the draw: each model's local attribution cache and global SHAP/PFI
    cache must hold exactly these instances. Logs the counts and returns rows of
    dataset, sample, class, n.
    """
    experiments = _dataset_experiment_paths(dataset_name, config_dir)
    seeds = {model: load_yaml(path).get("random_state", 42) for model, path in experiments}
    if len(set(seeds.values())) > 1:
        raise ValueError(f"{dataset_name} experiments do not share one random_state: {seeds}")
    _, _, _, _, x_test, y_test = _load_experiment_and_data(experiments[0][1], dataset_config_dir)
    random_state = seeds[experiments[0][0]]

    rows: list[dict[str, Any]] = []
    for sample, size in (("local", local_size), ("global", global_size)):
        sample_x = _class_balanced_sample(x_test, y_test, size, random_state=random_state)
        sample_y = y_test.loc[sample_x.index]
        _log_sample(f"{dataset_name} {sample} sample (total={size}, random_state={random_state})", sample_y)
        rows.extend(
            {"dataset": dataset_name, "sample": sample, "class": str(label), "n": int(count)}
            for label, count in sample_y.value_counts().sort_index().items()
        )
        rows.append({"dataset": dataset_name, "sample": sample, "class": "ALL", "n": len(sample_y)})

        if metrics_dir is None:
            continue
        for model, config_path in experiments:
            experiment_name = load_yaml(config_path)["name"]
            if sample == "local":
                caches = [Path(metrics_dir) / "agreement" / "attributions" / f"{experiment_name}.npz"]
            else:
                caches = [_global_importance_path(metrics_dir, experiment_name, method) for method in ("shap", "pfi")]
            for cache in caches:
                if not cache.exists():
                    logger.warning("%s: no cache at %s", experiment_name, cache)
                    continue
                cached_index = np.load(cache, allow_pickle=False)["instance_index"]
                if not np.array_equal(np.sort(cached_index), np.sort(sample_x.index.to_numpy())):
                    raise ValueError(f"{cache} was not computed on the {sample} sample drawn here")
    return pd.DataFrame(rows)
