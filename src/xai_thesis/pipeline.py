from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from xai_thesis.datasets import DATASET_REGISTRY
from xai_thesis.datasets.base import DatasetConfig, load_dataset_config
from xai_thesis.evaluation import (
    evaluate_classifier,
    plot_confusion_matrix,
    plot_feature_correlation,
    plot_label_distribution,
    plot_missing_values,
    plot_train_test_split_distribution,
)
from xai_thesis.explainability import (
    compute_permutation_importance,
    compute_shap_values,
    explain_instance_with_lime,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_waterfall,
    select_curated_instances,
)
from xai_thesis.models import PARAM_DISTRIBUTIONS, build_model, train_model
from xai_thesis.preprocessing import (
    build_preprocessing_pipeline,
    split_by_official_split_column,
    split_features_target,
    stratified_random_split,
    stratified_subsample,
)
from xai_thesis.utils import dump_yaml, ensure_directory, load_yaml

# xgboost's XGBClassifier has no class_weight constructor param at all (see models.py) - it
# only supports per-sample weights at fit() time. compute_sample_weight("balanced", y) computes
# the same per-class weighting class_weight="balanced" applies internally for random_forest/svm
# (and the same weights TorchMLPClassifier feeds to nn.CrossEntropyLoss), just passed through
# fit's sample_weight instead of set on the model itself. This is the only model that needs
# fit-time handling; all four reweight the objective, none resample.
_MODELS_REQUIRING_SAMPLE_WEIGHT = {"xgboost"}

_METRIC_EXTRACTORS = {
    "accuracy": lambda report: report["accuracy"],
    "precision_macro": lambda report: report["macro avg"]["precision"],
    "recall_macro": lambda report: report["macro avg"]["recall"],
    "f1_macro": lambda report: report["macro avg"]["f1-score"],
    # Every dataset in this project's catalog is multiclass (see evaluate_classifier()) - only
    # the macro variants exist. A prior binary Attack/Benign framing had its own
    # precision_attack/recall_attack/f1_attack/roc_auc/pr_auc entries here; removed 2026-08-27
    # since no config has requested them since that framing was dropped project-wide
    # (2026-07-31) and evaluate_classifier() no longer produces the report keys they read.
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
    # NSL-KDD ships with an official train/test split, used as-is. Every other dataset is
    # split with a stratified random partition (fixed seed), stratified on the label so each
    # class keeps the same share of rows in both splits - the conventional approach in the
    # XAI-for-IDS literature this project follows (e.g. Arreche et al. 2024). Each adapter's
    # clean() already drops its own non-feature columns (identifiers, capture timestamps);
    # only "source_file", the per-row loading tag every adapter adds, is dropped here.
    if config.name == "nsl_kdd":
        return split_by_official_split_column(data, split_column="split")
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


# svm's final fit on unsw_nb15 is intractable at the complete training-set size, re-confirmed
# 2026-08-28 after two separate attempts on the fully-corrected raw data (128k more rows, no
# more truncation/boundary duplicates than before): the first ran 18+ hours uncapped; a second
# attempt with max_iter=5000 (a fix that worked in principle - it does eventually stop) still
# ran 4h8m without finishing, worse than the run that prompted adding the cap in the first
# place. Lowering max_iter alone is not a reliable lever at this row count - each libsvm
# iteration itself is the expensive part, not just the iteration count - so the cap was removed
# in favor of this subsample, the one approach with an actual proven fast (~20 min) track
# record on this project's own history.
#
# cic_ids2017 is included here too, by explicit user request for consistency, even though its
# svm final fit had already completed successfully on the complete training set (see README.md
# for that run's numbers - now stale under this change and pending a retrain). The rationale:
# svm's final-fit methodology should be uniform across both datasets in this catalog rather
# than "whichever happened to finish in time gets the full set" - the same
# comparability-across-models argument that motivated removing this mechanism entirely a few
# sessions ago now argues for comparability-across-datasets instead, given unsw_nb15's svm
# cannot reliably train on its complete set at all.
#
# target_size raised 150_000->200_000 and min_per_class raised 200->500 on 2026-08-28, at
# explicit user request, after svm's first subsampled retrain (150k/200) came in below
# neural_network on macro-F1 (0.413 vs 0.449) on unsw_nb15. Checked directly before changing
# anything: raising target_size alone (150k->200k, min_per_class held at 200) adds rows almost
# entirely to the classes already well-represented (unsw_nb15's Benign alone absorbs ~90% of
# the increase) - every class sitting at the min_per_class floor gets zero additional rows
# either way, so target_size alone was very unlikely to move macro-F1, which weights every
# class equally including the weak ones. Raising min_per_class to 500 is the lever that
# actually changes anything for the floored classes: unsw_nb15's Analysis/Backdoor/Shellcode
# and cic_ids2017's Web Attack/PortScan/Bot all go from 200 to a full 500 rows each (verified
# directly - all six have well over 500 available in the complete training set). unsw_nb15's
# Worms (118 total) and cic_ids2017's Infiltration (24 total) are unaffected by either
# parameter - both floors already exceed their entire natural supply, so they're included in
# full regardless. Whether this actually closes the gap with neural_network is not guaranteed;
# it's the more targeted of the two available levers, not a proven fix.
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
        # random_forest / svm reweight via class_weight="balanced"; neural_network's
        # TorchMLPClassifier computes the same n/(k*count_c) weights from y inside fit().
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
    # RandomizedSearchCV/Pipeline re-slice any fit_param whose length matches len(x_search) per
    # CV fold automatically (a standard, well-supported sklearn mechanism - not special-cased
    # here), so this is safe to compute once over the full search set rather than per fold.
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


def _experiment_plot_dir(plots_dir: str | Path, dataset_name: str, experiment_name: str) -> Path:
    """Per-experiment plot folder, nested under its dataset (plots_dir/<dataset>/<model>_baseline/)
    rather than flat (plots_dir/<dataset>_<model>_baseline/) - keeps outputs/plots/ organized by
    dataset instead of mixing ~20 experiment folders and 4 dataset-level folders (label
    distribution, missing values, etc. - see generate_label_distribution and friends) in one flat
    namespace. Every experiment name in this project follows "<dataset>_<model>_baseline", so
    stripping the dataset prefix leaves just the model-identifying part as the subfolder name.
    """
    model_folder = experiment_name.removeprefix(f"{dataset_name}_")
    return Path(plots_dir) / dataset_name / model_folder


def generate_confusion_matrix(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    plots_dir: str | Path = "outputs/plots",
    normalize: bool = False,
) -> Path:
    """Plot a confusion matrix for an already-trained experiment's test predictions."""
    experiment, dataset_config, _, _, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )

    experiment_name = experiment["name"]
    model = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    predictions = model.predict(x_test)

    # Same per-experiment folder generate_explanations() writes into, so every plot for one
    # trained experiment (confusion matrix and explanations alike) lives in one place.
    output_path = _experiment_plot_dir(plots_dir, dataset_config.name, experiment_name) / "confusion_matrix.png"
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
    return plot_label_distribution(labels, output_path)


def generate_missing_value_summary(
    dataset_config_path: str | Path,
    *,
    plots_dir: str | Path = "outputs/plots",
) -> Path:
    """Plot per-column missing-value percentages for a processed dataset's combined
    train+test data. Imputation happens inside the sklearn Pipeline at training time (see
    preprocessing.py's build_preprocessing_pipeline), not before - so the processed CSVs on
    disk still carry whatever missingness survived each dataset adapter's own cleanup.
    """
    dataset_config = load_dataset_config(dataset_config_path)

    train_df = pd.read_csv(Path(dataset_config.processed_path) / "train.csv")
    test_df = pd.read_csv(Path(dataset_config.processed_path) / "test.csv")
    combined = pd.concat([train_df, test_df], ignore_index=True)

    output_path = Path(plots_dir) / dataset_config.name / "missing_values.png"
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

    output_path = Path(plots_dir) / dataset_config.name / "train_test_split.png"
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

    Sampled down to at most sample_size rows before computing correlations - this project's
    largest datasets have 1.5-2.3M processed rows, and pandas' .corr() over that many rows at
    80-140 columns is expensive for no real accuracy benefit (correlation structure is stable
    well below this sample size). Same subsampling-for-cost pattern already used for
    RandomizedSearchCV tuning on these same two datasets (see models.py/pipeline.py).
    """
    dataset_config = load_dataset_config(dataset_config_path)

    train_df = pd.read_csv(Path(dataset_config.processed_path) / "train.csv")
    features = train_df.drop(columns=[dataset_config.label_column])
    if len(features) > sample_size:
        features = features.sample(n=sample_size, random_state=42)

    output_path = Path(plots_dir) / dataset_config.name / "feature_correlation.png"
    return plot_feature_correlation(features, output_path, title=f"{dataset_config.name} feature correlation")


# (target_class, confused_with_class) per dataset - every dataset in the catalog is multiclass as
# of 2026-07-31. Informed by the confusion-matrix analysis: target_class is a class substantially
# confused with confused_with_class across every model trained, not just one - e.g. DAPT2020's
# Reconnaissance is misclassified as Benign by every model (10-75% of the time depending on model).
# Ties local explanations back to a concrete, already-established error pattern instead of an
# arbitrary/random instance choice.
_CLASS_OF_INTEREST: dict[str, tuple[str, str]] = {
    # Unlike this project's other multiclass datasets, UNSW-NB15's dominant confusion doesn't
    # funnel into Benign at all. Re-verified 2026-08-13/17 across three feature-set changes:
    # the raw-capture switch, dropping service+ct_ftp_cmd, then restoring service once
    # "improve performance" was set as the explicit priority over matching a reference paper's
    # unexplained preprocessing (service's removal had cost real macro-F1 - see
    # datasets/unsw_nb15.py; ct_ftp_cmd stayed dropped, that one evidence-based independent of
    # this policy). On the final feature set (service back in, ct_ftp_cmd out):
    # Analysis->Backdoor 7.8-62.5%, Backdoor->Analysis 10.9-49.7% - both directions present for
    # every model trained, ranges shifted per model but the two-way confusion itself hasn't
    # gone away across any of these feature-set changes. Analysis remains target_class: its
    # average recall is still the lower of the two (~41% vs ~47%), the harder-to-detect side
    # of this pair.
    "unsw_nb15": ("Analysis", "Backdoor"),
    "dapt2020": ("Reconnaissance", "Benign"),
    # R2L is confused with Benign more heavily than any other pair in this dataset, and by a
    # wide margin - confirmed across both models trained so far (82.5% of R2L misclassified
    # as Benign for gradient_boosting, 64.8% for logistic_regression).
    "nsl_kdd": ("R2L", "Benign"),
    # DoS/DDoS is misclassified as Benign by every model, the only confusion pattern that shows
    # up for all 5 - and it's also this dataset's largest attack category by far (64,319 of
    # 485,902 test rows), making a missed DoS/DDoS attack the single highest-volume false
    # negative this dataset can produce. Re-verified 2026-08-18 after Destination Port was
    # dropped (see configs/datasets/cic_ids2017.yaml - it was leaking each attack scenario's
    # fixed testbed target port, not generalizable behavior): the pair itself didn't change, but
    # the confusion got dramatically worse once that shortcut was removed - 68.9-89.3% now,
    # up from 6.3-13.2% on the old feature set. That jump is itself strong independent evidence
    # for how much the old numbers were propped up by the port leak rather than genuine
    # DoS/DDoS-vs-Benign separability.
    "cic_ids2017": ("DoS/DDoS", "Benign"),
}

# LIME's default discretize_continuous=True crashes on CIC-IDS2017's heavily-skewed columns
# (scipy truncnorm "scale must be positive" - confirmed reproducible, see explainability.py);
# verified safe for UNSW-NB15 specifically (30/30 test instances, no error) before enabling it
# there. This is orthogonal to binary vs. multiclass (it's about column skew, not class count),
# so UNSW-NB15's switch to multiclass doesn't change whether this is still safe there - it's
# restricted to unsw_nb15 here simply because that's the only dataset this was ever verified for.
_SAFE_TO_DISCRETIZE_LIME = {"unsw_nb15"}

# SHAP background size, and the number of test instances the global SHAP/PFI plots explain.
_SHAP_BACKGROUND_SIZE = 100
_GLOBAL_EXPLANATION_SAMPLE_SIZE = 200


def _class_balanced_sample(
    features: pd.DataFrame, target: pd.Series, total: int, *, random_state: int
) -> pd.DataFrame:
    """Sample ~total rows from features, spread evenly across the classes in target (a class
    with fewer rows than its even share contributes all of them).

    Used for the SHAP background so the explainer's reference distribution is the same
    ~uniform class mix for every model. A uniform random sample of this project's training
    data is 85-96% Benign (see the imbalance-handling note in README.md), which pins the
    permutation explainer's base value (svm, neural_network) to "an average Benign flow" and
    leaves it on a different footing from the tree models, whose TreeExplainer base value
    already reflects their class_weight/sample_weight-adjusted (near-balanced) internal prior.
    A class-balanced background puts svm/neural_network on that same near-balanced footing.
    """
    groups = [
        features.loc[index_block].sample(n=min(total // target.nunique() or 1, len(index_block)), random_state=random_state)
        for index_block in (target.index[target == label] for label in target.unique())
    ]
    return pd.concat(groups)


def generate_explanations(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    plots_dir: str | Path = "outputs/plots",
    sample_size: int = _GLOBAL_EXPLANATION_SAMPLE_SIZE,
) -> dict[str, Path]:
    """Generate SHAP, LIME, and permutation importance explanations for an already-trained
    experiment, saving plots under plots_dir/<dataset>/<model>_baseline/.

    The global SHAP bar/beeswarm and permutation-importance plots explain a sample_size-row
    sample of the test set drawn *stratified* (proportional to class prevalence), i.e. "what
    drives predictions on the traffic mix actually seen", not a per-class-balanced sample.
    Deliberate: it keeps the global plots representative of deployment conditions. The
    trade-off is that the rarest classes (UNSW-NB15 Worms, CIC-IDS2017 Infiltration) have
    near-zero rows in the sample, so their stacked segment in shap_bar.png is low-support -
    the per-class local explanations (curated instances, interview stimuli) are where rare
    classes are actually examined.

    Partial dependence plots (PDP) were removed entirely on 2026-08-27 (had been out of the
    thesis's explainability scope since 2026-07-31, and out of this function specifically
    since 2026-08-17) - explainability.py no longer has a plot_partial_dependence function at
    all, not just an unused one.
    """
    experiment, dataset_config, x_train, y_train, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )
    random_state = experiment.get("random_state", 42)
    experiment_name = experiment["name"]
    model_name = experiment["model"]

    if dataset_config.name not in _CLASS_OF_INTEREST:
        raise NotImplementedError(f"No class-of-interest mapping configured for dataset: {dataset_config.name}")
    target_class, confused_with_class = _CLASS_OF_INTEREST[dataset_config.name]

    pipeline = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    output_dir = ensure_directory(_experiment_plot_dir(plots_dir, dataset_config.name, experiment_name))
    class_names = list(pipeline.named_steps["model"].classes_)

    if len(x_test) > sample_size:
        sample_x, _, sample_y, _ = train_test_split(
            x_test, y_test, train_size=sample_size, stratify=y_test, random_state=random_state
        )
    else:
        sample_x, sample_y = x_test, y_test
    background_x = _class_balanced_sample(
        x_train, y_train, _SHAP_BACKGROUND_SIZE, random_state=random_state
    )

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

    pfi_path, _ = compute_permutation_importance(
        pipeline, sample_x, sample_y, output_dir / "permutation_importance.png", random_state=random_state
    )
    paths["permutation_importance"] = pfi_path

    paths.update(
        _generate_instance_explanations(
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
            discretize_continuous=dataset_config.name in _SAFE_TO_DISCRETIZE_LIME,
        )
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
    discretize_continuous: bool,
) -> dict[str, Path]:
    """SHAP waterfall + LIME plots for one curated (target_class, confused_with_class) instance
    pair - a correct classification and a misclassification of target_class as
    confused_with_class specifically. Shared by generate_explanations() (the thesis's own
    per-experiment _CLASS_OF_INTEREST pair) and generate_interview_stimuli() (multiple
    well-known classes per dataset, picked for interview relevance instead).
    """
    paths: dict[str, Path] = {}
    instances = select_curated_instances(
        pipeline, x_test, y_test, target_class, confused_with_class, random_state=random_state
    )
    if not instances:
        return paths

    instance_kinds = list(instances.keys())
    instance_indices = list(instances.values())
    instances_x = x_test.loc[instance_indices]
    instance_shap_values, instance_base_values, instances_transformed, feature_names = compute_shap_values(
        pipeline, model_name, instances_x, background_x, categorical_columns=categorical_columns
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
            categorical_columns=categorical_columns,
            random_state=random_state,
            discretize_continuous=discretize_continuous,
        )

    return paths


def _most_common_confusion(y_test: pd.Series, predictions: pd.Series, target_class: str) -> str | None:
    """The class target_class is most often predicted as, when the model gets it wrong -
    dynamically derived from real predictions rather than hand-picked, same "grounded in real
    evidence" principle as _CLASS_OF_INTEREST, just computed per-class on demand instead of
    curated once upfront. None if the model never misclassifies target_class in this test set
    (e.g. CIC-IDS2017's Bot for logistic_regression/neural_network - 100% recall via massive
    over-prediction rather than genuine skill, so no misclassified example exists at all).
    """
    is_target = y_test == target_class
    wrong = predictions[is_target & (predictions != target_class)]
    if len(wrong) == 0:
        return None
    return wrong.value_counts().idxmax()


# Well-known attack classes for interview stimulus curation - a deliberately different selection
# criterion from _CLASS_OF_INTEREST above (that one picks the class pair with the most
# significant cross-model *confusion*, for the thesis's own explainability-chapter analysis;
# this one picks classes a security practitioner would recognize by name, for showing to
# interview subjects) - a separate mechanism, not a replacement. Chosen 2026-08-17 with the
# user, verified against real per-class recall/precision/instance-availability before finalizing
# - ruled out: UNSW-NB15's Worms (only 130/33 train/test rows), Generic/Analysis/Backdoor/
# Shellcode (real classes, but not recognizable security terms to a general audience), DoS
# (well-known, but recall never exceeds 32% across any model - diagnosed as a genuine feature-
# overlap limitation shared with Analysis/Backdoor, not something a rate-based engineered
# feature fixed - see datasets/unsw_nb15.py); CIC-IDS2017's Web Attack (umbrella term, not a
# specific technique) and Bot (recall looks perfect for logistic_regression/neural_network only
# because they over-predict Bot almost everywhere - precision 0.011/0.015 - which also means
# zero misclassified examples exist for those two models, breaking the correct/misclassified
# stimulus pair entirely), Infiltration (only 28 training rows).
_INTERVIEW_STIMULUS_CLASSES: dict[str, list[str]] = {
    "unsw_nb15": ["Reconnaissance", "Exploits"],
    "cic_ids2017": ["DoS/DDoS", "PortScan", "Brute Force"],
}


def generate_interview_stimuli(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    plots_dir: str | Path = "outputs/plots",
) -> dict[str, dict[str, Path]]:
    """Generate SHAP waterfall + LIME stimuli for interview use: one correct + one misclassified
    instance per well-known class in _INTERVIEW_STIMULUS_CLASSES, for an already-trained
    experiment. Separate from generate_explanations()/_CLASS_OF_INTEREST (the thesis's own
    explainability-chapter analysis) - this exists purely to curate concrete cases to show
    interview subjects, picked for real-world recognizability rather than confusion severity.
    Saves under plots_dir/<dataset>/<model>_baseline/interview_<class>/, keyed in the return
    value by target class name, then by "shap_waterfall_correct"/"lime_correct"/etc. - a class
    maps to an empty dict if the model never misclassifies it in this test set (no stimulus pair
    possible, see _most_common_confusion).
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
    experiment_dir = _experiment_plot_dir(plots_dir, dataset_config.name, experiment_name)

    results: dict[str, dict[str, Path]] = {}
    for target_class in stimulus_classes:
        if target_class not in class_names:
            continue
        confused_with_class = _most_common_confusion(y_test, predictions, target_class)
        if confused_with_class is None:
            results[target_class] = {}
            continue
        output_dir = ensure_directory(experiment_dir / f"interview_{target_class.replace('/', '-')}")
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
            discretize_continuous=dataset_config.name in _SAFE_TO_DISCRETIZE_LIME,
        )
    return results
