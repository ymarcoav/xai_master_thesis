from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from xai_thesis.agreement import (
    AGREEMENT_METRIC_TITLES,
    AGREEMENT_METRICS,
    agreement_fraction,
    agreement_over_k,
    plot_agreement_heatmap,
)
from xai_thesis.datasets import DATASET_REGISTRY
from xai_thesis.datasets.base import DatasetConfig, load_dataset_config
from xai_thesis.evaluation import (
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
    _encode_lime_instance,
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
# attempt with a lowered max_iter=5000 (a fix that worked in principle - it does eventually
# stop) still ran 4h8m without finishing. Lowering max_iter alone is not a reliable lever at
# this row count - each libsvm iteration itself is the expensive part, not just the iteration
# count - so lowering the cap as the tractability fix was abandoned in favor of this subsample,
# the one approach with an actual proven fast (~20 min) track record on this project's own
# history. (models.py's generic max_iter=50_000 safety cap stays regardless - it only bounds
# worst-case time on a pathological fit, it is not the mechanism that makes the fit tractable.)
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

    output_path = _experiment_plot_dir(plots_dir, dataset_config.name, experiment_name) / "roc_curves.png"
    return plot_roc_curves(y_test, probabilities, list(model.classes_), output_path)


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


def _hardest_class_and_confusion(
    y_test: pd.Series, predictions: pd.Series
) -> tuple[str, str] | tuple[None, None]:
    """The class this trained model handles worst (lowest recall), and the class it is most
    often mistaken for.

    Replaces the old hand-maintained _CLASS_OF_INTEREST dict (one curated
    (target, confused_with) pair per dataset): that had to be re-verified by eye against every
    retrain, and after the stratified-split retune CIC-IDS2017's curated pair (DoS/DDoS ->
    Benign) had dissolved to ~0% confusion while still sitting in the config. Deriving both
    ends from this model's own test predictions keeps the explanation anchored to a real,
    current error pattern with no per-dataset config to go stale. Ties on recall break toward
    the better-supported class (more instances to curate from). Returns (None, None) if the
    model never misclassifies its lowest-recall class (e.g. a trivially separable toy model).
    """
    counts = y_test.value_counts()
    recall = {
        label: int(((y_test == label) & (predictions == label)).sum()) / int(counts[label])
        for label in counts.index
    }
    target_class = min(recall, key=lambda label: (recall[label], -counts[label]))
    confused_with_class = _most_common_confusion(y_test, predictions, target_class)
    if confused_with_class is None:
        return None, None
    return target_class, confused_with_class

# LIME's default discretize_continuous=True crashes on CIC-IDS2017's heavily-skewed columns
# (scipy truncnorm "scale must be positive" - confirmed reproducible, see explainability.py);
# verified safe for UNSW-NB15 specifically (30/30 test instances, no error) before enabling it
# there. This is orthogonal to binary vs. multiclass (it's about column skew, not class count),
# so UNSW-NB15's switch to multiclass doesn't change whether this is still safe there - it's
# restricted to unsw_nb15 here simply because that's the only dataset this was ever verified for.
_SAFE_TO_DISCRETIZE_LIME = {"unsw_nb15"}

# SHAP background size, and the number of test instances the global SHAP/PFI plots explain.
# The global sample is drawn class-balanced (see _class_balanced_sample), not stratified: a
# prevalence-proportional draw of this project's test sets is 85-96% Benign and leaves most
# attack classes with 0 rows (191/200 Benign on UNSW-NB15 at the old size of 200), so the
# global SHAP bar / PFI rankings were effectively a two-class problem. 500 class-balanced rows
# put every class in the ranking; classes with fewer test rows than their even share (e.g.
# CIC-IDS2017 Infiltration, ~11 rows) contribute all of them.
#
# The background is only consulted by the permutation path (svm, neural_network); the tree
# path ignores it. compute_shap_values passes it to shap.Explainer as-is (no further
# sub-sampling), so this is the exact reference distribution - keep it small, since svm's
# permutation SHAP cost scales with it (~90-410 s/instance at 50 rows). 50 divides evenly
# into both datasets' class counts (5/class x 10 on UNSW-NB15, 7/class x 7 on CIC-IDS2017).
_SHAP_BACKGROUND_SIZE = 50
_GLOBAL_EXPLANATION_SAMPLE_SIZE = 500


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
    metrics_dir: str | Path = "outputs/metrics",
    plots_dir: str | Path = "outputs/plots",
    sample_size: int = _GLOBAL_EXPLANATION_SAMPLE_SIZE,
) -> dict[str, Path]:
    """Generate SHAP, LIME, and permutation importance explanations for an already-trained
    experiment, saving plots under plots_dir/<dataset>/<model>_baseline/.

    The global SHAP bar/beeswarm and permutation-importance plots explain a sample_size-row
    *class-balanced* sample of the test set (_class_balanced_sample): ~even rows per class, a
    class with fewer test rows than its even share contributing all of them. This is a
    deliberate switch away from a stratified (prevalence-proportional) draw, which at
    sample_size=200 was 191/200 Benign on UNSW-NB15 and left 7 of 10 classes with zero rows -
    so the global rankings described a two-class problem. The trade-off is that the rankings no
    longer reflect deployment prevalence; they answer "what drives each class" with every class
    represented, and per-class prevalence is documented separately (label_distribution.png).

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

    pipeline = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")
    output_dir = ensure_directory(_experiment_plot_dir(plots_dir, dataset_config.name, experiment_name))
    class_names = list(pipeline.named_steps["model"].classes_)

    # Curated correct/misclassified instance explanations are anchored to the class this model
    # handles worst and the class it most often confuses it with, both derived from the model's
    # own test predictions (see _hardest_class_and_confusion). The global SHAP beeswarm targets
    # that same class.
    predictions = pd.Series(pipeline.predict(x_test), index=x_test.index)
    target_class, confused_with_class = _hardest_class_and_confusion(y_test, predictions)

    sample_x = _class_balanced_sample(x_test, y_test, sample_size, random_state=random_state)
    sample_y = y_test.loc[sample_x.index]
    background_x = _class_balanced_sample(
        x_train, y_train, _SHAP_BACKGROUND_SIZE, random_state=random_state
    )

    paths: dict[str, Path] = {}

    shap_values, base_values, x_transformed, feature_names = compute_shap_values(
        pipeline,
        model_name,
        sample_x,
        background_x,
        categorical_columns=dataset_config.categorical_columns,
        random_state=random_state,
    )
    paths["shap_bar"] = plot_shap_bar(
        shap_values, x_transformed, feature_names, class_names, output_dir / "shap_bar.png"
    )
    if target_class in class_names:
        target_index = class_names.index(target_class)
        paths["shap_beeswarm"] = plot_shap_beeswarm(
            shap_values, x_transformed, feature_names, target_index, output_dir / "shap_beeswarm.png"
        )

    # sample_x/sample_y are class-balanced (see above), so this PFI measures each feature's
    # macro-F1 drop under a *rebalanced* class distribution, not the natural 85-96%-Benign one.
    # Intentional: it matches the class-balanced global SHAP bar, and macro-F1 already weights
    # every class equally regardless of the sample's mix.
    pfi_path, _, pfi_result = compute_permutation_importance(
        pipeline, sample_x, sample_y, output_dir / "permutation_importance.png", random_state=random_state
    )
    paths["permutation_importance"] = pfi_path

    # The two rankings just plotted are exactly what the global agreement panel compares, so
    # cache them here rather than making generate_global_agreement recompute them (for svm that
    # is hours) - and so the two figures are guaranteed to describe the same numbers.
    column_order = [list(sample_x.columns).index(name) for name in feature_names]
    _save_global_rankings(
        _global_cache_path(metrics_dir, experiment_name, "balanced", sample_size),
        shap_ranking=np.abs(shap_values).sum(axis=2).mean(axis=0),
        permutation_mean=pfi_result.importances_mean[column_order],
        feature_names=feature_names,
        class_counts=sample_y.value_counts(),
    )

    if target_class is not None:
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
                predictions=predictions,
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
    predictions: pd.Series | None = None,
    discretize_continuous: bool,
) -> dict[str, Path]:
    """SHAP waterfall + LIME plots for one curated (target_class, confused_with_class) instance
    pair - a correct classification and a misclassification of target_class as
    confused_with_class specifically. Shared by generate_explanations() (target = the class
    this model handles worst, derived from its own predictions) and generate_interview_stimuli()
    (multiple well-known classes per dataset, picked for interview relevance instead). Pass
    predictions to reuse an already-computed full-test-set prediction.
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

    # xgboost's TreeExplainer is in raw-margin (log-odds) space on this shap/xgboost build;
    # every other model's SHAP values are probabilities (see compute_shap_values).
    waterfall_space = "log-odds (XGBoost raw margin)" if model_name == "xgboost" else "probability"

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
    derived from real predictions rather than hand-picked. Used by both
    _hardest_class_and_confusion (generate_explanations) and generate_interview_stimuli().
    None if the model never misclassifies target_class in this test set (e.g. a class with
    100% recall via massive over-prediction rather than genuine skill, so no misclassified
    example exists at all).
    """
    is_target = y_test == target_class
    wrong = predictions[is_target & (predictions != target_class)]
    if len(wrong) == 0:
        return None
    return wrong.value_counts().idxmax()


# Well-known attack classes for interview stimulus curation - a deliberately different selection
# criterion from generate_explanations() (which anchors on the class each model handles *worst*,
# for the thesis's own explainability-chapter error analysis); this one picks classes a security
# practitioner would recognize by name, for showing to interview subjects. Chosen 2026-08-17 with
# the user, re-checked 2026-09-01 against the stratified-split / retuned / Destination-Port-removed
# data - every listed class has >=1 correct and >=1 misclassified test instance for every model.
# Ruled out: UNSW-NB15's Worms (~120/51 rows), Generic/Analysis/Backdoor/Shellcode (not
# recognizable security terms to a general audience), DoS (well-known, but every model's recall
# on it is 6-38%); CIC-IDS2017's Web Attack (umbrella label folding SQLi/XSS/brute-force-login -
# a subject can't tell which technique an instance is), Infiltration (24 train rows).
# PortScan was dropped 2026-09-01: removing the leaky Destination Port column collapsed it under
# dedup from 158,930 native rows to 1,958 (~157k near-identical scan flows differed only in
# target port), so it is no longer a representative sample of scanning traffic. Bot (Ares botnet
# C2) replaced it - recognizable, and the one CIC-IDS2017 attack class every model still
# misclassifies (always -> Benign), so a genuine correct/misclassified pair exists for all four.
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
    """Generate SHAP waterfall + LIME stimuli for interview use: one correct + one misclassified
    instance per well-known class in _INTERVIEW_STIMULUS_CLASSES, for an already-trained
    experiment. Separate from generate_explanations()'s hardest-class analysis - this exists
    purely to curate concrete cases to show interview subjects, picked for real-world
    recognizability rather than by which class the model handles worst.
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
            predictions=predictions,
            discretize_continuous=dataset_config.name in _SAFE_TO_DISCRETIZE_LIME,
        )
    return results


# --- Cross-method agreement: SHAP vs LIME (Krishna et al. 2022) -----------------

# Instances the agreement metrics are averaged over. Class-balanced rather than
# stratified, so agreement is measured across attack types instead of on the
# 85-96%-Benign traffic mix a prevalence-proportional draw would give. The draw depends
# only on the test split and the seed, never on the model, so all four models are
# compared on the *identical* instances - which is what makes the result a statement
# about the two methods rather than about which rows each model happened to get.
_AGREEMENT_SAMPLE_SIZE = 60

# Top-k values the heatmaps sweep. Krishna et al. vary k the same way (their Figure 2, where
# agreement falls as k grows), but their k is a far larger share of the feature space than a
# naive copy of their numbers would give here: their headline k=5 is 5 of COMPAS's 7 features
# (71%), and they sweep to 100%, framing it as 25/50/75/100%. Five of this project's 40
# (UNSW-NB15) / 59 (CIC-IDS2017) columns is 8-12%. So the sweep runs to 20 (50% of UNSW-NB15,
# 34% of CIC-IDS2017) rather than stopping at 10, which would have squashed the whole figure
# into the top quarter of the ranking. It stops short of the feature count because feature
# agreement is 1.0 there by definition.
#
# k=1 is kept for the local panel only. There it is the mean of ~60 per-instance outcomes with
# a real standard error - "how often do the two methods even name the same single most
# important feature?", which is what an analyst reading the top driver of an alert cares about.
# On the global panel it would be a single ranking pair, so it could only ever read 0.0 or 1.0;
# that is a coin flip, not a measurement, so the global sweep starts at 5.
_AGREEMENT_K_VALUES = (1, 5, 10, 20)
_GLOBAL_AGREEMENT_K_VALUES = (5, 10, 20)

# Fixed row order for the heatmaps, so panels line up across datasets and metrics.
_AGREEMENT_MODEL_ORDER = ("random_forest", "xgboost", "neural_network", "svm")

# (dataset, model) pairs deliberately left out of both agreement panels, with the measured
# reason. Every agreement figure needs per-instance permutation SHAP for the non-tree models,
# and cic_ids2017's svm was measured at 408 s/instance (10-instance run at
# _SHAP_PERMUTATION_MULTIPLIER=50, 2026-09-09) - 6.8 h for the 60-instance local set and ~57 h
# for the 500-row global one, against ~90 s/instance for the same model on unsw_nb15, which is
# included. Excluded cells are dropped from the heatmap rows and named in its footnote, so the
# omission is visible in the figure rather than only in the text.
_AGREEMENT_EXCLUDED: dict[tuple[str, str], str] = {
    ("cic_ids2017", "svm"): "permutation SHAP measured at 408 s/instance",
}


def _partition_excluded(
    dataset_name: str, experiments: Sequence[tuple[str, Path]], models: Sequence[str] | None = None
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Split (model, config) pairs into those to compute and a note per model left out.

    Two reasons a model is left out: the permanent _AGREEMENT_EXCLUDED list (too expensive to
    compute at all, with the measured cost as its reason), and an explicit `models` filter,
    which the alternative-basis runs use to keep a methodological comparison off the models
    whose permutation SHAP costs hours.
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

    SHAP is computed in the transformed space and collapsed back onto the raw columns
    (proto/service/state summed from their one-hot dummies - see compute_shap_values);
    LIME already works in the raw space, minus any zero-variance column it drops. For
    this project's two datasets the name sets are identical (40 columns for UNSW-NB15, 59
    for CIC-IDS2017; nothing is dropped), so this is a guard rather than a real
    reconciliation - but a future dataset with a constant column would need it.
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

    Writes metrics_dir/agreement/attributions/<experiment>.npz with two
    [n_instances, n_features] signed matrices on a shared feature order, plus the feature
    names, the test-set row index, and the class each instance was explained for. Cached
    because svm's permutation SHAP costs ~90 s/instance on UNSW-NB15 and ~410 s on
    CIC-IDS2017; with the cache the heatmaps redraw in seconds.

    Both methods explain the same class for a given instance - the predict_proba argmax,
    which is what LIME's top_labels=1 selects - so the metrics compare two explanations of
    one prediction, as Krishna et al. define them.
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
    background_x = _class_balanced_sample(x_train, y_train, _SHAP_BACKGROUND_SIZE, random_state=random_state)

    shap_values, _, _, shap_feature_names = compute_shap_values(
        pipeline,
        model_name,
        sample_x,
        background_x,
        categorical_columns=categorical_columns,
        random_state=random_state,
    )

    # One LIME explainer for the whole loop - building it scans all of x_train.
    # num_features = every column so the surrogate weight vector comes back dense rather
    # than truncated to LIME's default top 8, which the top-k metrics need.
    lime_explainer, lime_predict_fn, lime_columns, lime_category_to_code = _build_lime_explainer(
        pipeline,
        x_train,
        class_names,
        categorical_columns=categorical_columns,
        random_state=random_state,
        discretize_continuous=dataset_config.name in _SAFE_TO_DISCRETIZE_LIME,
    )
    lime_rows: list[np.ndarray] = []
    explained_class: list[int] = []
    for instance_index in sample_x.index:
        explanation = lime_explainer.explain_instance(
            _encode_lime_instance(x_test.loc[instance_index], lime_columns, lime_category_to_code),
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
    averaged over the shared class-balanced instance set. Krishna et al. draw one
    method-by-method heatmap per model; this project has only two methods that produce
    per-instance attributions (SHAP and LIME - permutation importance is global-only and
    has no per-prediction explanation to compare), so the method pair is fixed and the
    panel varies model and k instead.

    Returns {"feature": path, "rank": path, "csv": path}; the CSV holds the same numbers
    plus their standard errors.
    """
    experiments, excluded_notes = _partition_excluded(
        dataset_name, _dataset_experiment_paths(dataset_name, config_dir), models
    )
    k_values = list(k_values)

    # The class-balanced instance draw in compute_explanation_attributions is seeded from each
    # experiment's own random_state, and the heatmap only means anything if every model is
    # scored on the *same* rows (that is what makes it a statement about SHAP vs LIME rather
    # than about which instances each model happened to get). Fail loudly if the configs
    # disagree rather than silently comparing models on different samples.
    seeds = {model: load_yaml(path).get("random_state", 42) for model, path in experiments}
    if len(set(seeds.values())) > 1:
        raise ValueError(
            f"agreement heatmaps need one shared random_state across all {dataset_name} "
            f"experiments (the instance draw depends on it); got {seeds}"
        )

    means = {metric: np.zeros((len(experiments), len(k_values))) for metric in AGREEMENT_METRICS}
    sems = {metric: np.zeros((len(experiments), len(k_values))) for metric in AGREEMENT_METRICS}
    rows: list[dict[str, Any]] = []

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
        for metric in AGREEMENT_METRICS:
            metric_means, metric_sems = agreement_over_k(shap_matrix, lime_matrix, k_values, metric)
            means[metric][row, :] = metric_means
            sems[metric][row, :] = metric_sems
            rows.extend(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "metric": metric,
                    "k": k,
                    "mean": round(float(mean), 4),
                    "sem": round(float(sem), 4),
                    "n_instances": int(shap_matrix.shape[0]),
                    "n_features": int(shap_matrix.shape[1]),
                }
                for k, mean, sem in zip(k_values, metric_means, metric_sems)
            )

    output_dir = ensure_directory(Path(plots_dir) / dataset_name)
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


# --- Cross-method agreement: SHAP vs permutation importance (global) -------------

# Sample bases the global ranking can be built on. "balanced" is what generate_explanations()
# uses for shap_bar.png / permutation_importance.png, so the default global agreement figure
# describes exactly the plots the thesis shows. "stratified" (prevalence-proportional) is kept
# because it was this project's basis until 2026-09-09, and the thesis needs to be able to
# reproduce the comparison that justified switching away from it - on unsw_nb15 a stratified
# 200-row draw is 191/200 Benign and leaves 7 of 10 classes with no rows at all.
_GLOBAL_AGREEMENT_BASES = ("balanced", "stratified")


def _global_ranking_sample(
    x_test: pd.DataFrame, y_test: pd.Series, size: int, *, basis: str, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Draw the test rows the global SHAP and permutation-importance rankings are built on."""
    if basis == "balanced":
        sample_x = _class_balanced_sample(x_test, y_test, size, random_state=random_state)
        return sample_x, y_test.loc[sample_x.index]
    if basis == "stratified":
        sample_x, _, sample_y, _ = train_test_split(
            x_test, y_test, train_size=size, stratify=y_test, random_state=random_state
        )
        return sample_x, sample_y
    raise ValueError(f"unknown basis {basis!r}; expected one of {list(_GLOBAL_AGREEMENT_BASES)}")


def _as_importance_vector(values: np.ndarray) -> np.ndarray:
    """Clip a global importance vector at zero before it is ranked.

    The agreement metrics rank features by |value|, which is what a *local* attribution needs
    (sign is direction, magnitude is strength). A global importance is already a magnitude:
    mean |SHAP| is non-negative by construction, and permutation importance goes slightly
    negative only when permuting a feature happened to improve the score, which means "no
    measurable importance", not "strongly important the other way". Without this clip a feature
    scoring -0.004 would outrank one scoring +0.001.
    """
    return np.clip(np.asarray(values, dtype=float), 0.0, None)


def _global_cache_path(metrics_dir: str | Path, experiment_name: str, basis: str, size: int) -> Path:
    """Where one experiment's global rankings live, keyed by the sample basis they came from."""
    cache_dir = ensure_directory(Path(metrics_dir) / "agreement" / "global")
    return cache_dir / f"{experiment_name}__{basis}{size}.npz"


def _save_global_rankings(
    cache_path: Path,
    *,
    shap_ranking: np.ndarray,
    permutation_mean: np.ndarray,
    feature_names: Sequence[str],
    class_counts: pd.Series,
) -> Path:
    """Write the two global importance vectors, on a shared feature order, plus the sample mix.

    Written from two places on purpose: compute_global_rankings() when the agreement figure is
    built on its own, and generate_explanations() as a by-product of drawing shap_bar.png and
    permutation_importance.png - which compute exactly these two rankings, over exactly these
    rows. Sharing the cache means the agreement figure describes the ranking the thesis actually
    plots rather than an independent re-run of it, and saves recomputing hours of svm SHAP.
    """
    np.savez(
        cache_path,
        shap=np.asarray(shap_ranking, dtype=float),
        permutation=np.asarray(permutation_mean, dtype=float),
        feature_names=np.array(list(feature_names)),
        class_names=np.array([str(label) for label in class_counts.index]),
        class_counts=class_counts.to_numpy(),
    )
    return cache_path


def compute_global_rankings(
    experiment_config_path: str | Path,
    *,
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    metrics_dir: str | Path = "outputs/metrics",
    size: int = _GLOBAL_EXPLANATION_SAMPLE_SIZE,
    basis: str = "balanced",
    overwrite: bool = False,
) -> Path:
    """Compute and cache one experiment's two global feature-importance rankings.

    Writes metrics_dir/agreement/global/<experiment>__<basis><size>.npz holding the mean |SHAP|
    pooled across classes and permutation_importance(...).importances_mean, both as
    [n_features] vectors on the same feature order, plus that sample's per-class row counts.

    Both are computed on the same rows, drawn per basis/size - at the defaults, exactly the
    sample generate_explanations() builds shap_bar.png and permutation_importance.png from.
    Cached because svm's permutation SHAP costs ~90 s/instance on unsw_nb15.
    """
    experiment, dataset_config, x_train, y_train, x_test, y_test = _load_experiment_and_data(
        experiment_config_path, dataset_config_dir
    )
    experiment_name = experiment["name"]
    cache_path = _global_cache_path(metrics_dir, experiment_name, basis, size)
    if cache_path.exists() and not overwrite:
        return cache_path

    random_state = experiment.get("random_state", 42)
    model_name = experiment["model"]
    pipeline = joblib.load(Path(models_dir) / f"{experiment_name}.joblib")

    sample_x, sample_y = _global_ranking_sample(
        x_test, y_test, size, basis=basis, random_state=random_state
    )
    background_x = _class_balanced_sample(x_train, y_train, _SHAP_BACKGROUND_SIZE, random_state=random_state)

    shap_values, _, _, feature_names = compute_shap_values(
        pipeline,
        model_name,
        sample_x,
        background_x,
        categorical_columns=dataset_config.categorical_columns,
        random_state=random_state,
    )
    # The same aggregation plot_shap_bar stacks: |SHAP| summed over classes, averaged over rows.
    shap_ranking = np.abs(shap_values).sum(axis=2).mean(axis=0)

    permutation = permutation_importance(
        pipeline, sample_x, sample_y, scoring="f1_macro", n_repeats=10, random_state=random_state, n_jobs=1
    )
    column_order = [list(sample_x.columns).index(name) for name in feature_names]

    return _save_global_rankings(
        cache_path,
        shap_ranking=shap_ranking,
        permutation_mean=permutation.importances_mean[column_order],
        feature_names=feature_names,
        class_counts=sample_y.value_counts(),
    )


def top_features(ranking: np.ndarray, feature_names: Sequence[str], k: int = 5) -> list[str]:
    """The k highest-importance feature names, most important first."""
    return [feature_names[index] for index in np.argsort(-_as_importance_vector(ranking))[:k]]


def generate_global_agreement(
    dataset_name: str,
    *,
    config_dir: str | Path = "configs/experiments",
    dataset_config_dir: str | Path = "configs/datasets",
    models_dir: str | Path = "outputs/models",
    metrics_dir: str | Path = "outputs/metrics",
    plots_dir: str | Path = "outputs/plots",
    k_values: Sequence[int] = _GLOBAL_AGREEMENT_K_VALUES,
    size: int = _GLOBAL_EXPLANATION_SAMPLE_SIZE,
    basis: str = "balanced",
    models: Sequence[str] | None = None,
    write_plots: bool | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Feature- and rank-agreement heatmaps between the two *global* rankings for one dataset.

    The same two Krishna et al. metrics the local (SHAP vs LIME) panels use, so the two are read
    on one scale; here the pair is mean |SHAP| pooled across classes versus permutation
    importance. Each cell is a single number rather than a mean over instances, because a global
    ranking is one vector per model - so there is no standard error to report here.

    Returns {"feature": path, "rank": path, "csv": path} on the default basis; an alternative
    basis returns {"csv": path} only (see write_plots). The CSV additionally records each
    model's top-5 features under both methods, which is what shows whether the ranking itself
    (not just the agreement score) moved when the sample basis changed.
    """
    experiments, excluded_notes = _partition_excluded(
        dataset_name, _dataset_experiment_paths(dataset_name, config_dir), models
    )
    k_values = list(k_values)

    values = {metric: np.zeros((len(experiments), len(k_values))) for metric in AGREEMENT_METRICS}
    rows: list[dict[str, Any]] = []

    for row, (model_name, config_path) in enumerate(experiments):
        cache_path = compute_global_rankings(
            config_path,
            dataset_config_dir=dataset_config_dir,
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            size=size,
            basis=basis,
            overwrite=overwrite,
        )
        cached = np.load(cache_path, allow_pickle=False)
        feature_names = [str(name) for name in cached["feature_names"]]
        shap_ranking = _as_importance_vector(cached["shap"])
        permutation_ranking = _as_importance_vector(cached["permutation"])

        for metric in AGREEMENT_METRICS:
            for column, k in enumerate(k_values):
                # agreement_fraction is per-point over [n_points, n_features]; a global ranking
                # is a single "point", so this is one row in and one value out.
                score = float(
                    agreement_fraction(shap_ranking[None, :], permutation_ranking[None, :], k, metric)[0]
                )
                values[metric][row, column] = score
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "basis": basis,
                        "size": int(cached["class_counts"].sum()),
                        "metric": metric,
                        "k": k,
                        "value": round(score, 4),
                        "n_features": len(feature_names),
                        "classes_present": len(cached["class_names"]),
                        "shap_top5": "; ".join(top_features(shap_ranking, feature_names)),
                        "permutation_top5": "; ".join(top_features(permutation_ranking, feature_names)),
                    }
                )

    # Exactly one standardised heatmap pair per dataset: the default basis only. The other
    # bases exist to justify that choice (see _GLOBAL_AGREEMENT_BASES) and are reported as
    # numbers - emitting a heatmap for each would bury the figure the thesis actually uses
    # under near-identical variants. Pass write_plots=True to override.
    default_basis = basis == "balanced" and size == _GLOBAL_EXPLANATION_SAMPLE_SIZE
    suffix = "" if default_basis else f"_{basis}{size}"
    if write_plots is None:
        write_plots = default_basis

    output_dir = ensure_directory(Path(plots_dir) / dataset_name)
    paths = {
        metric: plot_agreement_heatmap(
            values[metric],
            None,
            [model for model, _ in experiments],
            k_values,
            output_dir / f"global_agreement_{metric}{suffix}.png",
            metric=metric,
            title=(
                f"{AGREEMENT_METRIC_TITLES[metric]} - SHAP vs permutation importance, "
                f"{dataset_name} ({basis}-{size})"
            ),
            pair_label="SHAP vs permutation importance",
            footnote="; ".join(excluded_notes),
        )
        for metric in AGREEMENT_METRICS
    } if write_plots else {}

    csv_name = f"{dataset_name}_global_agreement{suffix}.csv"
    csv_path = ensure_directory(Path(metrics_dir) / "agreement") / csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths["csv"] = csv_path
    return paths
