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

Four intrusion detection datasets are supported: UNSW-NB15, CIC-IDS2017, DAPT2020, and NSL-KDD. Each
has a dataset adapter in `src/xai_thesis/datasets/` implementing dataset-specific cleanup (e.g.
deduplication, fixing known data-quality issues, dropping identifier/leakage columns) behind a shared
interface.

> **DAPT2020 is out of scope for the thesis as of 2026-07-31.** Its adapter, configs, trained models,
> and generated explanations all remain in place and fully functional - nothing was removed - but it is
> no longer part of the actual analysis the thesis reports on. Treat any DAPT2020-specific content here
> or elsewhere in this repo as legacy/reference material, not current thesis scope.

**Every dataset in this catalog is multiclass as of 2026-07-31** - UNSW-NB15 was the last binary one.
**UNSW-NB15** is multiclass over its native **10-category `attack_cat` scheme** (`Benign` + `Generic`,
`Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`) - the
standard categorization from the dataset's own source paper (Moustafa & Slay 2015), used essentially
as-is throughout UNSW-NB15 literature; unlike NSL-KDD/CIC-IDS2017, there's no common further
consolidation of this scheme. The dataset's native binary `label` column (0/1) is dropped in favor of
this. **CIC-IDS2017** is multiclass over **7 categories** (`Benign` + 6 attack families: `DoS/DDoS`,
`PortScan`, `Brute Force`, `Web Attack`, `Infiltration`, `Bot`), collapsed from its native ~15-way label -
this specific 7-category scheme matches Arreche et al. 2024 ("E-XAI: Evaluating Black-Box Explainable AI
Frameworks for Network Intrusion Detection", IEEE Access), confirmed against that paper's Figure 21
SHAP-plot legend (its prose description is internally inconsistent - it claims "six distinct attack
profiles" while naming seven, so the figure legend, not the prose, is the source of truth). `DoS Hulk`/
`DDoS`/`DoS GoldenEye`/`DoS slowloris`/`DoS Slowhttptest`/`Heartbleed` all fold into `DoS/DDoS`;
`FTP-Patator`/`SSH-Patator` fold into `Brute Force`; the three native `Web Attack - <subtype>` labels fold
into one `Web Attack` category. DAPT2020 is multiclass (its five kill-chain stages). NSL-KDD is kept
multiclass, but not at its native ~40-way specific attack-type granularity: it's grouped into the
conventional **DoS/Probe/R2L/U2R categories** (+ `Benign`) used throughout NSL-KDD literature -
structurally similar to DAPT2020's multi-stage setup.

UNSW-NB15's `Worms` category is severely data-scarce (127 training rows, 44 test rows) - by far the
rarest class in this dataset, in the same spirit as DAPT2020's `Data Exfiltration` or NSL-KDD's `U2R`.
Its dominant confusion pattern also doesn't funnel into `Benign` at all, unlike every other dataset here:
`Analysis` and `Backdoor` are confused with each other far more than with anything else, in both
directions, across every model trained - see `_CLASS_OF_INTEREST` in `src/xai_thesis/pipeline.py`.

CIC-IDS2017's native `Web Attack` labels carry a corrupted, non-ASCII separator byte baked into the raw
files themselves (confirmed via raw byte inspection: literal UTF-8 encoding of U+FFFD, not a decoding
artifact on this project's end) - the label-mapping code matches on `.startswith("Web Attack")` rather
than an exact string, so it's robust to however pandas happens to decode that byte.

One cleaning step was itself informed by the explainability analysis, not decided upfront: **DAPT2020**
drops `Src Port`/`Dst Port` alongside its IP/timestamp columns, since SHAP and permutation importance
surfaced suspicious port dominance (e.g. `Lateral Movement` driven by two fixed ports) that turned out to
be a testbed-specific identity leak rather than a generalizable signal. Retraining after the fix produced
measurably more coherent explanations (SHAP-vs-permutation-importance agreement improved), not just
better metrics.

**NSL-KDD**'s raw `KDDTrain+.txt`/`KDDTest+.txt` files ship with no header row and one undocumented
trailing column (a KDD-Cup-era per-record "difficulty level" score, not a traffic feature - confirmed by
checking actual row lengths against the dataset's documented 41 named attributes + class label, which
is one short); the adapter assigns column names explicitly and drops that column. Its native `class`
column (e.g. `neptune`, `buffer_overflow`) is mapped to the four attack categories via a lookup table
compiled from, and verified against, every distinct label actually present in *both* raw files - NSL-KDD's
test set deliberately introduces attack types absent from training (e.g. `mscan`, `httptunnel`,
`sqlattack`) to test generalization to novel attacks, so the mapping had to cover both, not just
training. An unrecognized label raises rather than silently falling back to `Benign`.

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

UNSW-NB15 and NSL-KDD both ship with their own official train/test split, used as-is. CIC-IDS2017 and
DAPT2020 (which ship without one) are split chronologically per source file and label, so every class
stays represented in both splits.

## Training

> **`logistic_regression` and `gradient_boosting` are out of scope for the thesis as of
> 2026-08-09.** Both remain fully implemented, trained, and explained - nothing was removed - but
> the thesis now reports on only three models: `neural_network`, `random_forest`, and `svm`. Treat
> any `logistic_regression`/`gradient_boosting` content here or elsewhere in this repo as
> legacy/reference material, not current thesis scope (same pattern as DAPT2020/PDP above).

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

All four datasets are severely class-imbalanced (NSL-KDD's `U2R` category has only 52 training rows and
67 test rows, against 67,343/9,711 `Benign`). `logistic_regression`, `gradient_boosting`,
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
different, later addition: `SVC` with a **linear kernel** and `probability=True`, matching an external
reference configuration exactly for those two settings. `probability=True` is kept despite the extra
internal Platt-scaling cost it adds on top of the kernel fit, to match that reference configuration
directly; scikit-learn 1.9 deprecates this in favor of wrapping with `CalibratedClassifierCV`, but the
parameter still works. As of 2026-08-09, `C` is tuned via `RandomizedSearchCV` like the other models
(the reference's own `C=0.5` is kept only as `--tune`'s starting point) - but with a deliberately
smaller search budget than the other models (`n_iter=8` instead of the usual default 30) called
directly via `tune_hyperparameters()` rather than the `scripts/train.py --tune` default, since `svm` is
measurably more expensive to fit than every other model in this catalog and the standard 30-iteration/
3-fold budget (90 fits) was projected to take many hours even on UNSW-NB15 alone.

`svm` is measurably more expensive than every other model in this catalog, at both training and
explanation time: an untuned fit took ~26 minutes on UNSW-NB15 (~101k training rows) and ~18 minutes on
DAPT2020 (~59k rows, but 5 classes means an internal one-vs-one decomposition into 10 calibrated binary
sub-models) - both slower than an earlier RBF-kernel attempt without `probability=True` (~5.5 minutes on
UNSW-NB15), confirming the calibration cost, not the kernel choice, dominates. Explanation generation is
slower too: `svm` has no structural SHAP explainer (same permutation-based fallback as `neural_network`),
and each permutation sample pays `SVC`'s comparatively expensive prediction cost, pushing SHAP generation
alone to 20-30 minutes per experiment for UNSW-NB15/NSL-KDD, and ~105 minutes for UNSW-NB15's later
10-class multiclass version (permutation SHAP cost scales with class count, not just row count).
NSL-KDD (~126k training rows, the most of any dataset `svm` had been trained on before CIC-IDS2017) took
only ~12 minutes to fit untuned - notably *faster* than either UNSW-NB15 or DAPT2020 despite having the
most rows, a reminder that kernel SVM training time tracks how separable the data is (how many support
vectors end up needed), not row count alone. `svm` is now also trained on **CIC-IDS2017** (~2M training
rows) as of 2026-08-09 - previously skipped as impractical; see the exact timing this actually took
below.

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

> **Partial dependence plots (PDP) are out of scope for the thesis's explainability analysis as of
> 2026-07-31.** The implementation, ICE-curve overlay, and generated plots all remain in place and
> fully functional - nothing was removed - but PDP output is no longer part of what the thesis actually
> reports on. Treat it as legacy/reference material, not current analysis scope; see the PDP bullet
> below for what it still produces.

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
- **Partial dependence plots** *(implemented, but out of scope for the thesis's analysis - see note
  above)* (`sklearn.inspection.PartialDependenceDisplay`, with individual conditional expectation curves
  overlaid): for the top numeric features by permutation importance. The
  ICE curves show one line per sampled instance alongside the usual average PDP line, so a feature whose
  effect is heterogeneous (pushing some instances toward the target class and others away from it) reads
  as heterogeneous, rather than averaging out to look flat or negligible.

The two curated instances per experiment (one correctly classified, one misclassified) are chosen from a
`(target_class, confused_with_class)` pair per dataset — not randomly — based on the cross-model confusion
patterns found via `scripts/confusion_matrix.py` (e.g. DAPT2020's `Reconnaissance` is confused with
`Benign` by every model trained so far, and CIC-IDS2017's `DoS/DDoS` — its largest attack category by far
— is confused with `Benign` by every model trained under its new 7-category scheme, 6.3-13.2% depending on
model). UNSW-NB15 is the exception to the "confused with Benign" pattern: `Analysis` and `Backdoor` are
confused with *each other* far more than with anything else, in both directions, across every model
trained (27.6-66.6% one way, 15.9-66.8% the other) — `Analysis` is picked as `target_class` since it has
the lower average recall of the two. See `_CLASS_OF_INTEREST` in `src/xai_thesis/pipeline.py`.

Plots are written to `outputs/plots/<experiment_name>/`.
