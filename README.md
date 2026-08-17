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

> **NSL-KDD is also out of scope for the thesis as of 2026-08-13.** Same treatment as DAPT2020: its
> adapter, configs, trained models, and generated explanations all remain in place and fully functional -
> nothing was removed - but it is no longer part of the actual analysis the thesis reports on. The thesis
> now covers only **UNSW-NB15 and CIC-IDS2017**. Treat any NSL-KDD-specific content here or elsewhere in
> this repo as legacy/reference material, not current thesis scope.

**Every dataset in this catalog is multiclass as of 2026-07-31** - UNSW-NB15 was the last binary one.
**UNSW-NB15** is multiclass over its native **10-category `attack_cat` scheme** (`Benign` + `Generic`,
`Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`) - the
standard categorization from the dataset's own source paper (Moustafa & Slay 2015), used essentially
as-is throughout UNSW-NB15 literature; unlike NSL-KDD/CIC-IDS2017, there's no common further
consolidation of this scheme. The dataset's native binary `label` column (0/1) is dropped in favor of
this.

**As of 2026-08-13, UNSW-NB15 is built from the full raw capture, not the pre-built 10%-sample.** The
`UNSW_NB15_training-set.csv`/`testing-set.csv` files used previously (a pre-cleaned, already-split 10%
sample published separately by the dataset's authors) are gone; the adapter now reads the 4 headerless
raw part files (`UNSW-NB15_1.csv`..`_4.csv`, 2,411,053 rows total - roughly 13x the old scale, comparable
to CIC-IDS2017) plus `NUSW-NB15_features.csv` for the column schema. This surfaced real raw-data quality
issues the pre-built files had already cleaned up: `attack_cat` is blank for benign rows (not the string
`"Normal"`) and inconsistently spelled `"Backdoor"`/`"Backdoors"` across the 4 files (normalized to
`"Backdoor"`); `is_ftp_login`/`ct_flw_http_mthd` are blank for ~54% of rows, verified to mean "not
applicable" rather than missing-at-random (filled with 0 directly, not left to the shared median
imputer); exactly one row across all 2.4M is truncated mid-record (dropped generically, via `label` not
being a valid 0/1, rather than a hardcoded row index). `srcip`/`dstip`/`sport`/`dsport` are dropped as
identifiers (not present in the old pre-built files either); the raw data also has no `rate` column (a
feature the original paper authors derived only for the pre-built files). The dataset now has no
official train/test split (the pre-built files provided one; the raw capture doesn't), so it's split
chronologically like CIC-IDS2017/DAPT2020 instead - see Preprocessing below.

**Two more UNSW-NB15 features were reconsidered on 2026-08-13/17, on top of the above: `service` and
`ct_ftp_cmd`.** Both were dropped outright on 2026-08-13. `service` followed Tai et al. 2024 ("A Robust
and Trustworthy Intrusion Detection System Using Adversarial Machine Learning and XAI", ATC), whose
UNSW-NB15 preprocessing drops it among 7 features with no stated rationale - checked directly against
this project's own data first: it's `"unknown"` (the raw `-`/blank placeholder) for 57.33% of rows
across both splits, a plausible unstated reason for their manual "not valuable" call (they explicitly
did not run automated feature selection for UNSW-NB15, unlike their CIC-IDS2017 pipeline, so this was a
human judgment call, not an algorithm's output). `ct_ftp_cmd` was a follow-up finding of this project's
own, not from that paper: verified to correlate with `is_ftp_login` at r=0.92 (near-redundant with a
feature already kept) and, when nonzero (only 1.95% of rows), to show no per-class differentiation -
its mean value is ~1.0-1.06 across every `attack_cat` with any nonzero rows at all.

**`service` was restored the same week once explicit policy was set: "improve performance" wins over
matching a reference paper's own unexplained preprocessing choices, when the two conflict.** Dropping
`service` had measurably cost real macro-F1 for `gradient_boosting`/`random_forest` (0.569→0.507,
0.566→0.524) for no performance reason of its own - once that tradeoff was surfaced and the priority
set, `service` went back into `categorical_columns` and all 5 models were retrained again. `ct_ftp_cmd`
stays dropped - that decision was evidence-based (redundant with `is_ftp_login`, no per-class signal),
independent of the paper-matching question, so the new policy doesn't reopen it. The current feature
set (41 columns) is a superset of the old pre-built-file feature set, identifiers and `ct_ftp_cmd`
aside.

**CIC-IDS2017** is multiclass over **7 categories** (`Benign` + 6 attack families: `DoS/DDoS`,
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

UNSW-NB15's `Worms` category is severely data-scarce (130 training rows, 33 test rows on the current
full-raw-capture data) - by far the rarest class in this dataset, in the same spirit as DAPT2020's `Data
Exfiltration` or NSL-KDD's `U2R`. Its dominant confusion pattern also doesn't funnel into `Benign` at all,
unlike every other dataset here: `Analysis`<->`Backdoor` remains a real two-way confusion for every model
trained on the final feature set (`service` restored, `ct_ftp_cmd` still dropped - see above), 7.8-62.5%
one way and 10.9-49.7% the other - present across the board, though the exact ranges have shifted with
each of this dataset's feature-set changes - see `_CLASS_OF_INTEREST` in `src/xai_thesis/pipeline.py`
for exact numbers.

CIC-IDS2017's native `Web Attack` labels carry a corrupted, non-ASCII separator byte baked into the raw
files themselves (confirmed via raw byte inspection: literal UTF-8 encoding of U+FFFD, not a decoding
artifact on this project's end) - the label-mapping code matches on `.startswith("Web Attack")` rather
than an exact string, so it's robust to however pandas happens to decode that byte.

**As of 2026-08-13, CIC-IDS2017's feature set was cut from ~79 to 61 columns**, driven by pairwise
correlation analysis on the processed data (not a plotted sample, which can be misleading - see below):
8 columns (`Bwd PSH/URG Flags`, all 6 `*_Bulk`/`Bulk_Rate` columns) are literally zero for every row in
the raw dataset, and 9 more are near-exact duplicates of another kept column (verified row-for-row, not
just via correlation - e.g. `Total Fwd Packets`/`Subflow Fwd Packets` differ in 0 of 2.83M rows) - see
`configs/datasets/cic_ids2017.yaml`'s `drop_columns` comment for the exact pairs. Also found in the same
pass: `Fwd Header Length`/`Bwd Header Length`/`min_seg_size_forward` carry physically impossible negative
values (as low as -3.22e10) in ~35 rows, almost certainly a CICFlowMeter integer-underflow bug - those
rows (not the columns, which are fine and meaningfully correlated once cleaned) are now dropped in
`clean()`. This was a genuine methodological trap worth noting: a correlation heatmap computed on a
sampled subset of rows initially flagged Fwd/Bwd Header Length as a duplicate pair (sample correlation
1.000), but the same computation on the full data gave 0.07 - the corrupted rows were extreme enough to
dominate whichever random sample happened to include them. Dropping the 17 columns above (all verified
either zero-variance or byte-identical to a kept twin) still required re-tuning `random_forest`
afterward - see Training below - since fewer total columns changes what `max_features="log2"` means at
each split, which briefly looked like a real regression on the rare `Bot` class before re-tuning fixed it.

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

NSL-KDD ships with its own official train/test split, used as-is - the only dataset in this catalog that
still does. CIC-IDS2017, DAPT2020, and (as of 2026-08-13) UNSW-NB15 ship without one, so they're split
chronologically per source file and label instead, so every class stays represented in both splits.
UNSW-NB15 sorts on `stime`, a Unix epoch integer verified genuinely chronological across its 4 raw part
files (22 Jan - 18 Feb 2015, files 1->4 in order).

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
parameter still works.

`svm` is measurably more expensive than every other model in this catalog, at both training and
explanation time: an untuned fit took ~26 minutes on UNSW-NB15 (~101k training rows) and ~18 minutes on
DAPT2020 (~59k rows, but 5 classes means an internal one-vs-one decomposition into 10 calibrated binary
sub-models) - both slower than an earlier RBF-kernel attempt without `probability=True` (~5.5 minutes on
UNSW-NB15), confirming the calibration cost, not the kernel choice, dominates. Explanation generation is
slower too: `svm` has no structural SHAP explainer (same permutation-based fallback as `neural_network`),
and each permutation sample pays `SVC`'s comparatively expensive prediction cost, pushing SHAP generation
alone to 20-30 minutes per experiment for UNSW-NB15/NSL-KDD's earlier binary versions, and ~105 minutes
for UNSW-NB15's later 10-class multiclass version (permutation SHAP cost scales with class count, not
just row count). NSL-KDD (~126k training rows) took only ~12 minutes to fit untuned - notably *faster*
than either UNSW-NB15 or DAPT2020 despite having the most rows (before CIC-IDS2017 was added), a
reminder that kernel SVM training time tracks how separable the data is (how many support vectors end up
needed), not row count alone.

**A hyperparameter set tuned for one feature space doesn't necessarily transfer to a smaller one.** When
CIC-IDS2017's feature set was cut from ~79 to 61 columns (see Datasets above), `random_forest`'s existing
tuned `model_params` (from before the cut) caused a real regression - `Bot`-class precision collapsed
from ~0.98 to 0.06 (macro-F1 0.909->0.827) - even though every dropped column was verified to carry zero
unique information. The cause: `max_features="log2"` is evaluated against the *current* column count, so
removing columns changes which random feature subsets get tried at every split throughout the forest,
regardless of whether those columns were informative. Re-running `RandomizedSearchCV` (not just re-fitting
with the old params) fixed it and then slightly improved on the original score (macro-F1 0.916). Treat any
column-set change to an already-tuned experiment as requiring a re-tune, not just a re-fit.

As of 2026-08-09, `C` is tuned via `RandomizedSearchCV` like the other models (the reference's own
`C=0.5` is kept only as `--tune`'s starting point). Getting this to run in reasonable time at all
required two real fixes, not just patience:

- **A deliberately smaller search budget** than the other models, called directly via
  `tune_hyperparameters()` rather than `scripts/train.py --tune`'s CLI (which doesn't expose these
  knobs): `n_iter=8` instead of the usual default 30 (the standard 30-iteration/3-fold budget, 90 fits,
  was projected to take many hours even on UNSW-NB15 alone), and for CIC-IDS2017 specifically, an
  explicit `subsample_size=40_000` passed to `tune_hyperparameters()` (well below the shared
  `200_000` default) so the search phase stays tractable - the final fit still trains on the complete,
  un-subsampled training set regardless, same as every other model.
- **`probability=False` during the search phase only.** `RandomizedSearchCV`'s `f1_macro` scoring calls
  `.predict()`, never `.predict_proba()`, so Platt-scaling calibration is pure overhead during search -
  `tune_hyperparameters()` in `pipeline.py` now builds the search-time pipeline with `probability=False`
  via a small model-params override (`_SEARCH_TIME_MODEL_PARAM_OVERRIDES`), never written back to the
  experiment config; the final fit in `run_experiment()` still gets `probability=True` (`svm`'s default),
  which is the only place it's actually needed, for the explanation pipeline's `predict_proba` calls.

**A genuine incident, not just slowness, surfaced while first tuning `svm` on UNSW-NB15**: an early
attempt (`C` candidates including 30 and 100) hung for **over 19 hours of CPU time** on what should have
been a multi-minute fit - almost certainly `class_weight="balanced"`'s extreme per-sample weights under
this project's severe imbalance causing libsvm's solver to essentially never converge for some `(C,
fold)` combination, the same class of failure already documented above for `gradient_boosting`'s
`l2_regularization=0`, just silent instead of producing an obviously-bad model. Fixed with two changes in
`models.py`: `SVC` now gets a hard `max_iter=50_000` safety cap (a fit that would have hung instead
stops early with a `ConvergenceWarning` and returns whatever solution it had reached - a far better
failure mode for an unattended search loop), and the `C` search grid was trimmed to drop the values most
likely to trigger this (`30`, `100`). Tuned results: UNSW-NB15 (`C=10`, search took 74 minutes) improved
macro-F1 from 0.313 to **0.360**; NSL-KDD (`C=10`, search took 30 minutes) reached macro-F1 **0.626**.
Both final fits still hit the `max_iter` cap without fully converging (visible via a
`ConvergenceWarning`) - a real, honestly-reported quality caveat, not hidden: the safety net trades a
small amount of potential model quality for bounded, predictable runtime, which given the alternative
(a silent multi-hour-or-worse hang) is the right trade for this project. **These UNSW-NB15 numbers
describe the old 10%-sample dataset** - see the next paragraph for the current, full-raw-capture data.

**UNSW-NB15's `svm` final fit is trained on a stratified subsample, not the complete training set -
unlike every other model/dataset combination in this catalog.** UNSW-NB15 jumped to ~1.9M training rows
when it switched from the pre-built 10%-sample to the full raw capture (see Datasets above) - the same
scale that already caused CIC-IDS2017's `svm` to hang for 24+ hours (next paragraph). Getting there also
hit its own, unrelated problem first: the *default* `tune_hyperparameters()` budget
(`subsample_size=200_000`, `n_jobs=2`) hung for 20+ hours tuning `gradient_boosting` on this same new
data (confirmed a genuine deadlock via live CPU-time sampling, not slowness - see the project's
HANDOVER.md for the full incident writeup); fixed by calling `tune_hyperparameters()` directly with an
explicit `subsample_size=40_000`, the same pattern CIC-IDS2017 already uses below - this unblocked
tuning for all 5 models. The final-fit problem is separate and was still real: `svm`'s final fit is the
one step in this whole pipeline that scales badly enough with row count to make "just tune smaller,
train on everything" infeasible (CIC-IDS2017's 24-hour non-completion below is the concrete proof). Fix:
`_FINAL_FIT_SUBSAMPLE` in `pipeline.py` stratified-subsamples `svm`'s final training set to 150,000 rows
(`min_per_class=200`, comfortably above every class's full row count so no class gets thinned further -
UNSW-NB15's rarest, `Worms`, has 130) before fitting, for `unsw_nb15`/`cic_ids2017` specifically - a
deliberate, disclosed methodological deviation, not an oversight. Current UNSW-NB15 `svm` result on the
final feature set (`service` restored, `C=10` re-tuned): macro-F1 **0.393** (~19min for the final fit) -
essentially flat versus the intermediate without-`service` result (0.390), unlike the tree-ensemble
models, which recovered more macro-F1 once `service` came back.

**CIC-IDS2017's `svm` is tuned but not yet trained.** `configs/experiments/cic_ids2017_svm_baseline.yaml`
exists and its `C` search completed (`C=10`, search took only ~8 minutes thanks to an explicit
`subsample_size=40_000` override well below the shared 200k default - CIC-IDS2017 is ~2M rows). The
final fit on the complete, un-subsampled training set is the genuinely open question: an attempt was
started and left running unattended for 24+ hours (confirmed still actively computing throughout, not
hung - `max_iter` bounds iteration *count*, but at this row count each individual iteration itself costs
far more than on any other dataset here, so the cap doesn't translate to a comparable bound on
wall-clock time) before being interrupted by an unrelated session restart with no checkpoint to resume
from - `outputs/models/cic_ids2017_svm_baseline.joblib` does not exist yet. Deliberately not retried
within this session; picking this back up is a deferred, explicit decision, not an oversight.

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

Plots are written to `outputs/plots/<dataset_name>/<model>_baseline/confusion_matrix.png`.

Label class distribution for a dataset, or all datasets:

```powershell
python scripts/label_distribution.py --dataset unsw_nb15
python scripts/label_distribution.py  # all datasets
```

Plots are written to `outputs/plots/<dataset_name>/label_distribution.png` - as of 2026-08-17,
`outputs/plots/` is organized per dataset first (`outputs/plots/<dataset_name>/`), holding
dataset-level plots like this one directly and one `<model>_baseline/` subfolder per trained
experiment for its confusion matrix and explanation plots - not a flat mix of ~20 experiment
folders and 4 dataset folders sharing one namespace, as it was before.

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
  LIME can be compared directly on identical cases. Rendered as a three-panel figure reproducing LIME's
  `show_in_notebook()` layout as a static image (`explain_instance_with_lime_full` in
  `explainability.py`) - prediction probabilities on the left (predicted class highlighted), feature
  weights for the predicted class in the center (green supports it, red opposes it, strongest at top),
  and a feature-value table on the right, row-aligned and color-matched to the weights panel so which
  value drove which bar is visible at a glance. `show_in_notebook`/`save_to_file` only produce an
  HTML+D3 widget with no static-image path, which doesn't work in this project's headless
  (`matplotlib.use("Agg")`) pipeline, so this rebuilds the same three panels directly from the
  explanation object's own data (`predict_proba`, `as_list`, `local_exp`) instead. The single-panel
  `explanation.as_pyplot_figure()` version LIME ships by default is still available as
  `explain_instance_with_lime` (both share one `_build_lime_explanation()` helper so the explanation
  itself is only computed once regardless of which rendering is requested) but isn't currently called
  anywhere in the pipeline.
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
model). UNSW-NB15 is the exception to the "confused with Benign" pattern: `Analysis`↔`Backdoor` remains a
real two-way confusion for every model trained on the final feature set (`service` restored,
`ct_ftp_cmd` still dropped — 7.8-62.5% one way, 10.9-49.7% the other) — `Analysis` is still picked as
`target_class`, since its average recall is still the lower of the two (~41% vs ~47%). See
`_CLASS_OF_INTEREST` in `src/xai_thesis/pipeline.py`.

Plots are written to `outputs/plots/<dataset_name>/<model>_baseline/`.
