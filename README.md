# xai_master_thesis

Code for the master thesis titled "Evaluating Explainable AI Methods for Intrusion Detection Systems: From Model Explanations to Security Decisions".

## Structure

- `configs/datasets/`: one YAML per dataset (raw/processed paths, label column, columns to drop, categorical columns).
- `configs/experiments/`: one YAML per trained model (dataset, model type, hyperparameters, metrics to report).
- `data/`: raw and processed datasets. Data files are ignored by Git.
- `src/xai_thesis/`: reusable project code (dataset adapters, preprocessing, models, evaluation, explainability, pipeline orchestration).
- `scripts/`: command-line entry points for inspection, preprocessing, training, and analysis.
- `notebooks/`: exploratory analysis and thesis figures in progress.
- `outputs/`: generated models (`models/`), metrics (`metrics/`), and plots (`plots/`).
- `tests/`: focused tests for shared code.

## Setup

```powershell
pip install -e ".[dev]"
```

## Datasets

Three intrusion detection datasets are supported: UNSW-NB15, CIC-IDS2017, and DAPT2020. Each has a
dataset adapter in `src/xai_thesis/datasets/` implementing dataset-specific cleanup (e.g. deduplication,
fixing known data-quality issues, dropping identifier/leakage columns) behind a shared interface.

CIC-IDS2017 and UNSW-NB15 use **binary classification** (`Benign`/`Attack`), matching prior literature
that reports binary results for these two datasets specifically; UNSW-NB15's binary label uses the
dataset's own native `label` column rather than deriving it from the multiclass `attack_cat`. DAPT2020
remains multiclass (its five kill-chain stages), since there's no equivalent binary convention for it in
the literature.

One cleaning step was itself informed by the explainability analysis, not decided upfront: **DAPT2020**
drops `Src Port`/`Dst Port` alongside its IP/timestamp columns, since SHAP and permutation importance
surfaced suspicious port dominance (e.g. `Lateral Movement` driven by two fixed ports) that turned out to
be a testbed-specific identity leak rather than a generalizable signal. Retraining after the fix produced
measurably more coherent explanations (SHAP-vs-permutation-importance agreement improved), not just
better metrics.

### Dataset inspection

Inspect raw dataset files before implementing preprocessing:

```powershell
python scripts/inspect_datasets.py
```

For a quick sampled check:

```powershell
python scripts/inspect_datasets.py --max-rows 1000
```

Reports are written to `outputs/metrics/dataset_inspection/`.

### Preprocessing

Load, clean, and split one or all datasets into `data/processed/<dataset>/{train,test}.csv`:

```powershell
python scripts/preprocess.py --dataset unsw_nb15
python scripts/preprocess.py  # all datasets
```

UNSW-NB15 uses its official train/test split; CIC-IDS2017 and DAPT2020 (which ship without one) are
split chronologically per source file and label, so every class stays represented in both splits.

## Training

Each experiment config in `configs/experiments/` names a dataset and a model
(`logistic_regression`, `neural_network`, `gradient_boosting`, `random_forest`, or `svm`). Train one:

```powershell
python scripts/train.py --experiment configs/experiments/unsw_nb15_gradient_boosting_baseline.yaml
```

Add `--tune` to search for hyperparameters first (`RandomizedSearchCV`, stratified CV) and write the
best ones back into the experiment's `model_params` before training on the full training set:

```powershell
python scripts/train.py --experiment configs/experiments/unsw_nb15_gradient_boosting_baseline.yaml --tune
```

Trained pipelines (preprocessing + model) are saved to `outputs/models/<experiment_name>.joblib`,
and metrics to `outputs/metrics/<experiment_name>.json`.

All three datasets are severely class-imbalanced. `logistic_regression`, `gradient_boosting`,
`random_forest`, and `svm` all use `class_weight="balanced"`; `neural_network` (`MLPClassifier`) has no
`class_weight`/`sample_weight` support in scikit-learn, so its training data is resampled instead
(`balance_classes` in `preprocessing.py` oversamples rare classes and undersamples common ones to the
median class count, applied only to the training fold, both during hyperparameter search and the final
fit).

`random_forest` was originally dropped from the model catalog (shares `gradient_boosting`'s
tree-ensemble family and `TreeExplainer`, consistently showed the highest cross-model SHAP agreement
with it of any model pair, and underperformed it on 2 of 3 datasets) - re-added by explicit request
despite that redundancy concern still applying. It gets the same `class_weight="balanced"` and
`TreeExplainer` treatment as `gradient_boosting`.

An earlier `svm` (`LinearSVC`) was dropped for substantially worse macro-F1 than the other models,
replaced by `logistic_regression` as the catalog's linear model. The current `svm` entry is a
different, later addition: `SVC` with a **linear kernel**, `C=0.5`, and `probability=True`, matching
an external reference configuration exactly rather than being tuned via this project's usual
`RandomizedSearchCV` - there is deliberately no hyperparameter search space for it. `probability=True`
is kept despite the extra internal Platt-scaling cost it adds on top of the kernel fit, to match that
reference configuration directly; scikit-learn 1.9 deprecates this in favor of wrapping with
`CalibratedClassifierCV`, but the parameter still works. `svm` is measurably more expensive than every
other model in this catalog, at both training and explanation time: fitting it took ~26 minutes on
UNSW-NB15 (~101k training rows) and ~18 minutes on DAPT2020 (~59k rows, but 5 classes means an
internal one-vs-one decomposition into 10 calibrated binary sub-models) - both slower than an earlier
RBF-kernel attempt without `probability=True` (~5.5 minutes on UNSW-NB15), confirming the calibration
cost, not the kernel choice, dominates. Explanation generation is slower too: `svm` has no structural
SHAP explainer (same permutation-based fallback as `neural_network`), and each permutation sample pays
`SVC`'s comparatively expensive prediction cost, pushing SHAP generation alone to 20-30 minutes per
experiment. Given this, `svm` has not been trained on CIC-IDS2017 (~2M training rows) - extrapolating
from the measured UNSW-NB15/DAPT2020 timings, a single fit there would plausibly take hours to days.

## Analysis

Confusion matrix (absolute counts, colored on a symmetric-log scale so cells of very different
magnitude all stay visually distinguishable under severe class imbalance; the color scale's top end is
anchored to the largest value *excluding* the single most extreme cell, so one dominant majority-class
cell doesn't compress every other cell's shade into the same dark band) for a trained experiment, or all
experiments in `configs/experiments/`:

```powershell
python scripts/confusion_matrix.py --experiment configs/experiments/unsw_nb15_gradient_boosting_baseline.yaml
python scripts/confusion_matrix.py  # all experiments
```

Plots are written to `outputs/plots/<experiment_name>/confusion_matrix.png`.

Label class distribution for a dataset, or all datasets:

```powershell
python scripts/label_distribution.py --dataset unsw_nb15
python scripts/label_distribution.py  # all datasets
```

Plots are written to `outputs/plots/<dataset_name>/label_distribution.png`.

## Explainability

Four techniques are implemented in `src/xai_thesis/explainability.py`, generated per trained experiment
(or all of them) via:

```powershell
python scripts/explain.py --experiment configs/experiments/unsw_nb15_gradient_boosting_baseline.yaml
python scripts/explain.py  # all experiments
```

- **SHAP**: `TreeExplainer` for `gradient_boosting` and `random_forest`, `LinearExplainer` for
  `logistic_regression`, and a permutation-based `Explainer` for `neural_network` and `svm` (no fast
  structural explainer exists for sklearn MLPs or kernel SVMs - this path is markedly slower for both,
  and slower still for `svm` specifically, since its own predictions are more expensive to compute).
  Produces a global bar plot (mean |SHAP value| per feature, stacked by class), a global beeswarm
  for the dataset's class of interest, and local waterfall plots for two curated test instances.
- **LIME** (`LimeTabularExplainer`): local explanations for the same two curated instances, so SHAP and
  LIME can be compared directly on identical cases.
- **Permutation feature importance** (`sklearn.inspection.permutation_importance`): run on the full
  pipeline, so importances are reported in terms of the original feature names, not the one-hot-expanded
  ones.
- **Partial dependence plots** (`sklearn.inspection.PartialDependenceDisplay`, with individual
  conditional expectation curves overlaid): for the top numeric features by permutation importance. The
  ICE curves show one line per sampled instance alongside the usual average PDP line, so a feature whose
  effect is heterogeneous (pushing some instances toward the target class and others away from it) reads
  as heterogeneous, rather than averaging out to look flat or negligible.

The two curated instances per experiment (one correctly classified, one misclassified) are chosen from a
`(target_class, confused_with_class)` pair per dataset — not randomly — based on the cross-model confusion
patterns found via `scripts/confusion_matrix.py` (e.g. DAPT2020's `Reconnaissance` is confused with
`Benign` by every model trained so far). See `_CLASS_OF_INTEREST` in `src/xai_thesis/pipeline.py`.

Plots are written to `outputs/plots/<experiment_name>/`.
