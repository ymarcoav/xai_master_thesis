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
    explain_instance_with_lime_full,
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
    # Only well-defined for the binary Attack/Benign datasets - see evaluate_classifier().
    "precision_attack": lambda report: report["Attack"]["precision"],
    "recall_attack": lambda report: report["Attack"]["recall"],
    "f1_attack": lambda report: report["Attack"]["f1-score"],
    "roc_auc": lambda report: report["roc_auc"],
    "pr_auc": lambda report: report["pr_auc"],
    # Multiclass counterparts to roc_auc/pr_auc, only well-defined for DAPT2020 - see evaluate_classifier().
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


# DAPT2020's raw Timestamp column is kept through clean() (unlike its other identity/leakage
# columns - see datasets/dapt2020.py) specifically so the split below can sort by it, then
# drops it immediately afterward. This was added after checking DAPT2020's actual raw
# Timestamps: row order within a source file is NOT chronological (25-40% of adjacent rows
# go backwards in time in the files checked), so without sorting, a "last N rows per group"
# split does not actually hold out later traffic, despite the split being described as
# chronological. CIC-IDS2017's raw files in this project have no Timestamp column at all, so
# its split stays positional - there is nothing to sort by - and should be described as such,
# not as verified chronological. UNSW-NB15's stime is already a numeric Unix epoch (no format
# string needed - see split_chronologically_by_source); verified genuinely chronological across
# its 4 raw part files (Jan 22 - Feb 18 2015, files 1->4 in order, matching the dataset's
# documented two-session capture window).
_CHRONOLOGICAL_SPLIT_TIMESTAMP: dict[str, tuple[str, str | None]] = {
    "dapt2020": ("Timestamp", "%d/%m/%Y %I:%M:%S %p"),
    "unsw_nb15": ("stime", None),
}


def _split_dataset(config: DatasetConfig, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # NSL-KDD ships with an official train/test split, used as-is. UNSW-NB15 no longer does
    # (unlike the pre-built training-set/testing-set files this project used previously, the
    # full raw capture it's now built from has no predefined split), so it's split
    # chronologically per source file instead, same as CIC-IDS2017/DAPT2020 - see
    # preprocessing.py for why.
    if config.name == "nsl_kdd":
        return split_by_official_split_column(data, split_column="split")
    timestamp_column, timestamp_format = _CHRONOLOGICAL_SPLIT_TIMESTAMP.get(config.name, (None, None))
    return split_chronologically_by_source(
        data,
        source_column="source_file",
        label_column=config.label_column,
        timestamp_column=timestamp_column,
        timestamp_format=timestamp_format,
    )


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


# Model params applied only to the search-time pipeline built inside tune_hyperparameters(),
# never written back to the experiment config - the final fit in run_experiment() always uses
# the model's normal defaults/config. svm's probability=True is skipped during search
# specifically: RandomizedSearchCV's f1_macro scoring only ever calls .predict(), never
# .predict_proba(), so Platt-scaling calibration is pure overhead there - and this project's
# own measurements found it dominates SVC's fit cost more than the kernel choice itself (an
# RBF fit without it took ~5.5 min on UNSW-NB15 vs. ~26 min for the linear-kernel fit with it,
# same data - see models.py). This alone is expected to cut the search phase by roughly the
# same factor.
_SEARCH_TIME_MODEL_PARAM_OVERRIDES: dict[str, dict[str, Any]] = {
    "svm": {"probability": False},
}


def _build_pipeline(experiment: dict[str, Any], dataset_config: DatasetConfig, x_train: pd.DataFrame) -> Pipeline:
    random_state = experiment.get("random_state", 42)
    preprocessing = build_preprocessing_pipeline(x_train, categorical_columns=dataset_config.categorical_columns)
    model = build_model(experiment["model"], random_state=random_state, **experiment.get("model_params", {}))
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])


def _fit_pipeline_with_post_preprocessing_resampling(
    experiment: dict[str, Any],
    dataset_config: DatasetConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Pipeline:
    """Fit preprocessing on the true training distribution first, then resample only the
    already-transformed features before fitting the model - unlike resampling x_train/y_train
    before the combined preprocessing+model Pipeline is built and fit, which lets the
    resampled (duplicated) distribution leak into the imputer's learned median and, for
    UNSW-NB15, which rare categories cross the one-hot encoder's frequency-bucketing
    threshold. MinMaxScaler's min/max are unaffected either way (duplicating rows doesn't
    change the range), so this only changes neural_network's fitted preprocessing, not the
    other two models', which never resample.

    Reassembles a normal Pipeline from the two separately-fitted pieces afterward (a
    supported scikit-learn pattern: predict/predict_proba only call transform/predict on each
    already-fitted step, they don't require Pipeline.fit() to have been called), so callers
    can treat the result identically to one built via _build_pipeline + train_model.
    """
    preprocessing = build_preprocessing_pipeline(x_train, categorical_columns=dataset_config.categorical_columns)
    x_train_transformed = preprocessing.fit_transform(x_train)
    if hasattr(x_train_transformed, "toarray"):
        x_train_transformed = x_train_transformed.toarray()
    x_train_transformed = pd.DataFrame(x_train_transformed, columns=preprocessing.get_feature_names_out())

    x_resampled, y_resampled = balance_classes(
        x_train_transformed, y_train.reset_index(drop=True), random_state=random_state
    )

    model = build_model(experiment["model"], random_state=random_state, **experiment.get("model_params", {}))
    # .to_numpy(): fitting on a DataFrame would make the model remember the transformed,
    # ColumnTransformer-prefixed column names (e.g. "numeric__dur") as its own
    # feature_names_in_, which every other model-fitting path in this project avoids (the
    # normal Pipeline.fit() path always hands the model a bare array, since ColumnTransformer's
    # own output isn't a DataFrame) - matching that avoids a spurious "X does not have valid
    # feature names" warning on every future pipeline.predict() call with this model.
    train_model(model, x_resampled.to_numpy(), y_resampled)

    return Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])


# svm's final fit is intractable at these two datasets' full training-set size: CIC-IDS2017's
# ~2M-row attempt ran 24+ hours without completing (see README.md), and UNSW-NB15 hit the same
# risk after switching to the full raw capture (~1.9M rows) and was deferred rather than repeat
# that incident. Unlike every other model/dataset combination in this catalog, svm's final fit
# on these two is trained on a stratified subsample instead of the complete training set - a
# deliberate, disclosed methodological deviation, not an oversight. min_per_class (200) is set
# well above each dataset's rarest class's full count (CIC-IDS2017's Infiltration: 28 rows;
# UNSW-NB15's Worms: 130 rows), so stratified_subsample's "at least min_per_class, or all of
# them if fewer" rule keeps every rare class fully intact rather than thinning it down further
# to a handful of rows or less.
_FINAL_FIT_SUBSAMPLE: dict[tuple[str, str], tuple[int, int]] = {
    ("svm", "unsw_nb15"): (150_000, 200),
    ("svm", "cic_ids2017"): (150_000, 200),
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

    if experiment["model"] in _MODELS_REQUIRING_RESAMPLING:
        pipeline = _fit_pipeline_with_post_preprocessing_resampling(
            experiment, dataset_config, x_train, y_train, random_state
        )
    else:
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

    # Unlike run_experiment()'s final fit (see _fit_pipeline_with_post_preprocessing_resampling),
    # this resamples before the combined preprocessing+model Pipeline is built, so each
    # RandomizedSearchCV fold's own preprocessing still fits on already-resampled data, and a
    # duplicated oversampled row can land on both sides of a fold split. Properly closing this
    # here would need per-fold-safe resampling inside the CV loop (e.g. imbalanced-learn's
    # Pipeline, which supports a sampler step that only resamples during fit, not transform) -
    # not implemented, since it would add a new dependency for a search-time-only concern: it
    # can only bias which hyperparameters get selected, not the final reported model or metrics,
    # which run_experiment() computes independently on the corrected fit.
    if experiment["model"] in _MODELS_REQUIRING_RESAMPLING:
        x_search, y_search = balance_classes(x_search, y_search, random_state=random_state)

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

    # Same per-experiment folder generate_explanations() writes into, so every plot for one
    # trained experiment (confusion matrix and explanations alike) lives in one place.
    output_path = Path(plots_dir) / experiment_name / "confusion_matrix.png"
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

    # Per-dataset folder (distinct namespace from the per-experiment folders used for
    # confusion matrices/explanations, since a dataset name is never equal to one of its own
    # experiment names) - this plot isn't tied to any one trained model.
    output_path = Path(plots_dir) / dataset_config.name / "label_distribution.png"
    return plot_label_distribution(labels, output_path, title=f"{dataset_config.name} label distribution")


# (target_class, confused_with_class) per dataset - every dataset in the catalog is multiclass as
# of 2026-07-31. Informed by the confusion-matrix analysis: target_class is a class substantially
# confused with confused_with_class across every model trained, not just one - e.g. DAPT2020's
# Reconnaissance is misclassified as Benign by every model (10-75% of the time depending on model).
# Ties local explanations back to a concrete, already-established error pattern instead of an
# arbitrary/random instance choice.
_CLASS_OF_INTEREST: dict[str, tuple[str, str]] = {
    # Unlike this project's other multiclass datasets, UNSW-NB15's dominant confusion doesn't
    # funnel into Benign at all. Re-verified 2026-08-13 after switching UNSW-NB15 from the
    # pre-built 10%-sampled training-set/testing-set files to the full raw capture (2.4M rows,
    # ~13x the old scale - see datasets/unsw_nb15.py): Analysis->Backdoor is still a top-2
    # confusion for every model trained (18.2-69.4%), but the picture is noisier at this scale
    # than it was on the old data - DoS and Reconnaissance now also confuse heavily with
    # Backdoor/Analysis/Exploits/Shellcode for some models, and the reverse direction
    # (Backdoor->Analysis) is no longer consistently dominant (0-36.7%, actually 0% for
    # neural_network). Analysis remains target_class regardless: its recall is still clearly
    # the lower of the two on the new data (~31% vs ~56% average across models), the
    # harder-to-detect side of this pair.
    "unsw_nb15": ("Analysis", "Backdoor"),
    "dapt2020": ("Reconnaissance", "Benign"),
    # R2L is confused with Benign more heavily than any other pair in this dataset, and by a
    # wide margin - confirmed across both models trained so far (82.5% of R2L misclassified
    # as Benign for gradient_boosting, 64.8% for logistic_regression).
    "nsl_kdd": ("R2L", "Benign"),
    # DoS/DDoS is misclassified as Benign by every model trained under the new 7-category
    # multiclass scheme (6.3-13.2% depending on model), the only confusion pattern that shows
    # up for all 4 - and it's also this dataset's largest attack category by far (64,319 of
    # 485,902 test rows), making a missed DoS/DDoS attack the single highest-volume false
    # negative this dataset can produce.
    "cic_ids2017": ("DoS/DDoS", "Benign"),
}

# LIME's default discretize_continuous=True crashes on CIC-IDS2017's heavily-skewed columns
# (scipy truncnorm "scale must be positive" - confirmed reproducible, see explainability.py);
# verified safe for UNSW-NB15 specifically (30/30 test instances, no error) before enabling it
# there. This is orthogonal to binary vs. multiclass (it's about column skew, not class count),
# so UNSW-NB15's switch to multiclass doesn't change whether this is still safe there - it's
# restricted to unsw_nb15 here simply because that's the only dataset this was ever verified for.
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
            pipeline,
            sample_x,
            numeric_top_features[:pdp_top_n],
            target_class,
            output_dir / "partial_dependence.png",
            random_state=random_state,
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
            paths[f"lime_{kind}"] = explain_instance_with_lime_full(
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
