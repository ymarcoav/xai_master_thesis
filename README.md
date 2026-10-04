# xai_master_thesis

Code for the master thesis titled "Evaluating Explainable AI Methods for Intrusion Detection Systems: From Model Explanations to Security Decisions".

## Structure

- `configs/datasets/`: one YAML per **active** dataset (raw/processed paths, label column, columns to drop, categorical columns).
- `configs/experiments/`: one YAML per **active** trained model (dataset, model type, hyperparameters, metrics to report).
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

`torch` (used only by the `neural_network` model - see below) is a listed dependency. The
project needs the CPU build only; if `pip` pulls a large CUDA wheel, install it explicitly
first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

### Pinned environment

`pyproject.toml` lists dependencies unpinned, for development. **`requirements.lock` pins the exact
environment that produced everything in `outputs/`** - every model, metric and explanation figure -
and is what the thesis's numbers should be reproduced against:

```powershell
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock
.venv/Scripts/python -m pip install -e . --no-deps
```

Python 3.14.0 (CPython, Windows 11). The versions that actually change explanation results:

| package | pinned | why it matters |
| --- | --- | --- |
| `shap` | 0.52.0 | `_SHAP_PERMUTATION_MULTIPLIER` is set from `npermutations = max_evals // (2 * n_features + 1)`, the arithmetic in this version's `shap/explainers/_permutation.py:169`. A version changing that formula, or the `max_evals=500` default it compensates for, changes how many permutations `svm`/`neural_network` SHAP actually runs. Also the version whose `PermutationExplainer` accepts `seed=`, and whose `TreeExplainer` rejects `model_output="probability"` for `xgboost` - which is why that model's attributions are in log-odds space. |
| `lime` | 0.2.0.1 | `LimeTabularExplainer`'s `num_samples=5000` default and `sqrt(n_features) * 0.75` kernel width are this version's; both feed the local agreement panel. |
| `scikit-learn` | 1.9.0 | `permutation_importance` (PFI plots and global top-10), `SVC` + Platt scaling, the preprocessing pipeline. |
| `xgboost` | 3.4.1 | Pairs with `shap` for the `TreeExplainer` output-space behaviour above. |
| `torch` | 2.13.0+cpu | `TorchMLPClassifier`; the CPU build, seeded via `torch.use_deterministic_algorithms`. |
| `pandas` | 3.0.4 | `DataFrame.applymap` was removed here - the reason `agreement.py`'s port of the reference `agreement_fraction` uses `.astype(str)`. |
| `numpy` / `scipy` | 2.4.6 / 1.18.0 | `rankdata(method="dense")` and the SEM used by the agreement metrics. |

## Datasets

Two intrusion detection datasets are supported: UNSW-NB15 and CIC-IDS2017. Each
has a dataset adapter in `src/xai_thesis/datasets/` implementing dataset-specific cleanup (e.g.
deduplication, fixing known data-quality issues, dropping identifier/leakage columns) behind a shared
interface.

### Cleaning steps at a glance

The complete list of what each adapter's `clean()` does, in order, with counts on the current raw
capture. The thesis's methodology chapter summarizes these and refers here for the detail.

**UNSW-NB15** (`src/xai_thesis/datasets/unsw_nb15.py`; 2,540,044 raw rows, 49 columns):

1. Column names attached from `NUSW-NB15_features.csv` (the four part files have no header).
2. Numeric columns coerced to numbers; unparseable entries would become NaN for the median imputer.
   None occur: the only non-numeric entries are the blanks of step 4.
3. Rows whose native `label` is not `0`/`1` are dropped (guards against truncated exports; drops 0
   rows on the current files).
4. `is_ftp_login` and `ct_flw_http_mthd` blanks (56.3% / 53.1% of rows, concentrated in non-FTP /
   non-HTTP flows, i.e. "not applicable") are set to 0, not imputed.
5. `is_ftp_login` values above 1 (156 rows of `4`, 30 of `2`) are clipped to 1, its documented binary
   range.
6. `service`'s `-` placeholder (1,246,395 rows) becomes the category `unknown`. (The code also maps an
   empty string, which does not occur in the current files.)
7. `attack_cat` is whitespace-stripped, blanks become `Benign`, and `Backdoors` becomes `Backdoor`.
8. The 8 `drop_columns` are removed (identifiers, timestamps, `ct_ftp_cmd`, native `label`); the
   feature screen removes nothing (`screen_drop_columns` is empty). 40 features remain.
9. Exact duplicate rows are removed (ignoring the source-file tag): 2,026,609 rows remain.

**CIC-IDS2017** (`src/xai_thesis/datasets/cic_ids2017.py`; 2,830,743 raw rows, 79 columns):

1. Leading/trailing spaces stripped from column names; `Fwd Header Length.1`, a byte-for-byte copy of
   `Fwd Header Length`, is dropped.
2. `inf` values (`Flow Bytes/s`, `Flow Packets/s` on zero-duration flows) become NaN for the median
   imputer.
3. Rows with physically impossible negative values are dropped: 2,891 with a negative `Flow Duration` or
   `*IAT*` value (a known CICFlowMeter timestamp bug), and 35 more with a negative `Fwd/Bwd Header
   Length` or `min_seg_size_forward` (an integer underflow) - 2,926 rows in total.
4. Native labels are consolidated into 7 classes (`_ATTACK_CATEGORY`; the three `Web Attack` variants,
   whose separator is a corrupted non-ASCII byte, are matched by prefix).
5. `Destination Port` (`drop_columns`) and the 17 `screen_drop_columns` (8 constant, 9 near-duplicates;
   see `scripts/feature_screen.py`) are removed. 59 features remain.
6. Exact duplicate rows are removed (ignoring the source-file tag): 2,231,126 rows remain.

> **DAPT2020 and NSL-KDD were removed from the project on 2026-10-01** (out of thesis scope since
> 2026-07-31 / 2026-08-13, archived 2026-08-17): their adapters, tests, configs and the NSL-KDD-only
> official-split code path are gone, and their archived data and outputs were deleted the same day.

**Both datasets are multiclass** - UNSW-NB15 was the last binary one, switched on 2026-07-31.
**UNSW-NB15** is multiclass over its native **10-category `attack_cat` scheme** (`Benign` + `Generic`,
`Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`) - the
standard categorization from the dataset's own source paper (Moustafa & Slay 2015), used essentially
as-is throughout UNSW-NB15 literature; unlike CIC-IDS2017, there's no common further
consolidation of this scheme. The dataset's native binary `label` column (0/1) is dropped in favor of
this.

**As of 2026-08-13, UNSW-NB15 is built from the full raw capture, not the pre-built 10%-sample.** The
`UNSW_NB15_training-set.csv`/`testing-set.csv` files used previously (a pre-cleaned, already-split 10%
sample published separately by the dataset's authors) are gone; the adapter now reads the 4 headerless
raw part files (`UNSW-NB15_1.csv`..`_4.csv`) plus `NUSW-NB15_features.csv` for the column schema. This
surfaced real raw-data quality issues the pre-built files had already cleaned up: `attack_cat` is blank
for benign rows (not the string `"Normal"`) and inconsistently spelled `"Backdoor"`/`"Backdoors"` across
the 4 files (normalized to `"Backdoor"`); `is_ftp_login`/`ct_flw_http_mthd` are blank for ~54% of rows,
verified to mean "not applicable" rather than missing-at-random (filled with 0 directly, not left to the
shared median imputer). `srcip`/`dstip`/`sport`/`dsport` are dropped as identifiers (not present in the
old pre-built files either); the raw data also has no `rate` column (a feature the original paper authors
derived only for the pre-built files). The dataset now has no official train/test split (the pre-built
files provided one; the raw capture doesn't), so it gets the same stratified random split as
CIC-IDS2017 - see Preprocessing below.

**2026-08-28: the raw capture itself was found and fixed to be incomplete, not just differently-scaled
from the old pre-built files.** The initially-downloaded `UNSW-NB15_4.csv` was truncated mid-record at
its very last line and the whole set totaled only 2,411,053 rows, ~129k short of the dataset's own
officially published total (2,540,044). Re-downloading file 4 fixed the truncation; separately, each
consecutive pair of raw files was found to overlap by exactly one byte-identical row at the file boundary
(confirmed at all three 1↔2/2↔3/3↔4 boundaries) - the true, undocumented reason the per-file line counts
were 700,001 rather than the documented 700,000. Both raw-file issues are now fixed directly (the
duplicate boundary row removed from the start of files 2-4), and the raw capture now totals exactly
2,540,044 rows, matching the official figure exactly. This project's own row-level dedup step already
collapsed the boundary duplicates in the final processed data before this was even noticed (dedup ignores
which source file a row came from by design), so fixing the raw files changed nothing about model
inputs/outputs - it only corrected the raw file counts and the truncated final row, which previously
silently dropped ~129k real rows' worth of training/test data.

**Two more UNSW-NB15 features were reconsidered on 2026-08-13/17, on top of the above: `service` and
`ct_ftp_cmd`.** Both were dropped outright on 2026-08-13. `service` followed Tai et al. 2024 ("A Robust
and Trustworthy Intrusion Detection System Using Adversarial Machine Learning and XAI", ATC), whose
UNSW-NB15 preprocessing drops it among 7 features with no stated rationale - checked directly against
this project's own data first: it's `"unknown"` (the raw `-` placeholder) for 57.33% of rows
across both splits, a plausible unstated reason for their manual "not valuable" call (they explicitly
did not run automated feature selection for UNSW-NB15, unlike their CIC-IDS2017 pipeline, so this was a
human judgment call, not an algorithm's output). `ct_ftp_cmd` was a follow-up finding of this project's
own, not from that paper: verified to be near-redundant with the retained `is_ftp_login` (r≈0.92 on the
deduplicated data, r≈0.85 on the raw capture before dedup) and, when nonzero (only ~2% of rows), to show
no per-class differentiation - its mean is ~1 (1.00-1.20) across every `attack_cat` with any nonzero rows
at all. It is also documented as type-inconsistent (whitespace-padded in the raw files - Tai et al. 2024).

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
into one `Web Attack` category.

UNSW-NB15's `Worms` category is severely data-scarce - **171 rows in the processed data (120 train / 51
test)** under the stratified split; the "174" figure sometimes quoted elsewhere is the pre-dedup raw
count. By far the rarest class in this dataset. For scale, `Benign` is **95.6%** of both the UNSW-NB15 train and test splits
(CIC-IDS2017: 84.9% `Benign`, `Infiltration` just 24 train / 11 test rows). UNSW-NB15's dominant error
pattern also doesn't funnel into `Benign` at all: `Analysis`↔`Backdoor` is a
strong two-way confusion for the tree and neural-network models (~44-69% each way; the linear SVM
separates them better) - the explainability chapter's hardest-class analysis lands here.

CIC-IDS2017's native `Web Attack` labels carry a corrupted, non-ASCII separator byte baked into the raw
files themselves (confirmed via raw byte inspection: literal UTF-8 encoding of U+FFFD, not a decoding
artifact on this project's end) - the label-mapping code matches on `.startswith("Web Attack")` rather
than an exact string, so it's robust to however pandas happens to decode that byte.

**As of 2026-08-13, CIC-IDS2017's feature set was cut from ~79 to 61 columns**, driven by pairwise
correlation and row-for-row comparison on the full dataset (~2.83M rows, not a plotted sample, which can
be misleading - see below): 8 columns (`Bwd PSH/URG Flags`, all 6 `*_Bulk`/`Bulk_Rate` columns) are
literally zero for every row in the raw dataset, and 9 more are near-duplicates of another kept column -
each repeats a kept column's value in all but at most 20 of the 2.23M cleaned rows (six identical in
every row; `RST`/`ECE Flag Count` differ in 3 rows, r = 0.9978). The criterion is value identity, not a
correlation threshold: pairs that merely correlate are kept even above r = 0.998 (e.g. `Flow Duration`/
`Fwd IAT Total`, r = 0.9985, differ in 77% of rows). These 17 columns are listed under
`screen_drop_columns` in `configs/datasets/cic_ids2017.yaml`, and `scripts/feature_screen.py` reproduces
the screen (see Dataset inspection below). Also found in the same
pass: `Fwd Header Length`/`Bwd Header Length`/`min_seg_size_forward` carry physically impossible negative
values (as low as -3.22e10) in ~35 rows, almost certainly a CICFlowMeter integer-underflow bug - those
rows (not the columns, which are fine and meaningfully correlated once cleaned) are now dropped in
`clean()`. This was a genuine methodological trap worth noting: a correlation heatmap computed on a
sampled subset of rows initially flagged Fwd/Bwd Header Length as a duplicate pair (sample correlation
1.000), but the same computation on the full data gave 0.07 - the corrupted rows were extreme enough to
dominate whichever random sample happened to include them. Dropping the 17 columns above (all verified
either zero-variance or a near-exact duplicate of a kept twin) still required re-tuning `random_forest`
afterward - see Training below - since fewer total columns changes what `max_features="log2"` means at
each split, which briefly looked like a real regression on the rare `Bot` class before re-tuning fixed it.

**A port-leak pattern surfaced in CIC-IDS2017, 2026-08-17/18** - user-flagged this time
("Destination Port... regularly appears in the explainability plots, but I believe it is an identifier
that needs to be removed"), then verified directly before acting on it: `Destination Port` was carrying
34% of `random_forest`'s total feature importance alone (>2.5x the next-highest feature, the #1 SHAP
feature for every class), and per-class port concentration was near-total - `DoS/DDoS` 99.999% port 80,
`Web Attack` 100% port 80, `Infiltration` 100% port 444, `Brute Force` 81%/19% port 21/22, `Bot` 55% port
8080 - because the CICIDS2017 testbed launched each attack scenario against one fixed target service, the
a testbed-identity leak, not generalizable traffic behavior. A broader per-class value-concentration scan across the other 57 columns found nothing else
structurally identifier-like (no Source/Destination IP, Flow ID, or Protocol column exists in this raw
CICIDS2017 variant at all); the many other high-concentration columns it flagged turned out to be
genuine attack-traffic behavior (e.g. `PortScan`'s near-empty single-packet probes), not artifacts, and
were deliberately left in place. Dropped from `configs/datasets/cic_ids2017.yaml`'s `drop_columns`
(feature set now 60 columns, was 61), and - learning from this project's own `max_features="log2"`
column-count-sensitivity incident above - all 5 CIC-IDS2017 models (the 3 in-scope plus the 2 archived
`gradient_boosting`/`logistic_regression` copies) were re-tuned, not just re-fit, and had their confusion
matrices, explanations, and interview stimuli fully regenerated. The macro-F1 cost was large and is the
honest, expected result, not a bug: `random_forest` 0.916→0.605, `svm` 0.518→0.366, `neural_network`
0.510→0.354, `logistic_regression` 0.527→0.368, `gradient_boosting` 0.881→0.753 (the tree ensembles held
up best, still noticeably degraded). Immediately after the drop, `DoS/DDoS`→`Benign` confusion jumped from 6.3-13.2% to 68.9-89.3% -
strong independent evidence that the old numbers were substantially propped up by the port leak. (On
the later stratified-split re-tune that confusion resolved to ~0%: with an i.i.d. split and no port
leak, `DoS/DDoS` is cleanly separable.)

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

Constant-column and near-duplicate screen (the datasets chapter's feature screen), checked against each
config's `screen_drop_columns`:

```powershell
python scripts/feature_screen.py
```

Each dataset is cleaned as for training but with its `screen_drop_columns` kept, and every numeric
feature is screened (`src/xai_thesis/screening.py`): a column is removed if it is constant, or if it
repeats a retained column's value in all but at most 0.01% of rows. Pearson correlation (|r| >= 0.99)
only nominates candidate pairs; the row-by-row comparison decides. The evidence for every constant
column and candidate pair goes to `outputs/metrics/dataset_inspection/<dataset>/feature_screen.csv`, and
the script exits with status 1 if the screen and the config disagree. Current result: CIC-IDS2017's 8
constant columns and 9 duplicates (at most 20 differing rows) match the config; UNSW-NB15 has none (its
closest pair, `swin`/`dwin`, differs in 2,540 rows).

### Preprocessing

Load, clean, and split one or all datasets into `data/processed/<dataset>/{train,test}.csv`:

```powershell
python scripts/preprocess.py --dataset unsw_nb15
python scripts/preprocess.py  # all datasets
```

Neither dataset ships with an official train/test split, so both get a **stratified random split**
(`stratified_random_split` in `preprocessing.py`): a single `train_test_split` with a fixed
`random_state`, `stratify` on the label so every class keeps the same share of rows in both
partitions. This is the conventional choice in the XAI-for-IDS literature this project's methodology
otherwise follows (e.g. Arreche et al. 2024, whose multiclass label grouping the dataset adapters
already match), and it is simpler to state and reproduce than a temporal hold-out. Each adapter's
`clean()` already drops its own non-feature columns, including the `stime`/`Timestamp` capture times
(kept only long enough for dedup to collapse a flow recorded twice at different times); the split
just drops the per-row `source_file` loading tag every adapter adds.

**The default ratio is 70/30** (`test_fraction` default in `preprocessing.py`), matching the ratio
used by most of those reference papers; it is not motivated by anything in this project's own data.

> **Switched from a per-(source_file, label) chronological/positional hold-out to this stratified
> random split on 2026-08-31** (explicit user request: match the conventional XAI-for-IDS protocol).
> All datasets were reprocessed, all models re-tuned and retrained, and all explanation artifacts
> regenerated under the new split; the numbers below are current.

## Training

> **`logistic_regression` and `gradient_boosting` were removed from this project entirely on
> 2026-08-24** (explicit user request). Both had been flagged out-of-scope-for-the-thesis since
> 2026-08-09 and archived (models/metrics/plots/configs moved to `outputs/_archive/`/
> `configs/_archive/`) since 2026-08-17, then deleted outright rather than kept around unused:
> their model factory entries and tuning grids (`models.py`), every archived config/model/
> metrics/plot file, and now-dead explanation code (`logistic_regression`'s dedicated
> `shap.LinearExplainer` branch) are all gone, not just hidden. Historical numbers/results
> elsewhere in this document that reference these two models are kept as-is - they describe real
> findings from when the models existed, not a claim that they still do.
>
> **`xgboost` (added 2026-08-24) is in scope for the thesis** (corrected 2026-08-27 - earlier
> text here called it out-of-scope-but-functional, which was wrong as of this date). The thesis
> now reports on four models: `neural_network`, `random_forest`, `svm`, and `xgboost`. Unlike
> every other model here, `XGBClassifier` has no `class_weight` constructor parameter at all -
> see the imbalance-handling paragraph below for how it's balanced instead.

Each experiment config in `configs/experiments/` names a dataset and a model (`neural_network`,
`random_forest`, `svm`, or `xgboost`). Train one:

```powershell
python scripts/train.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml
```

Add `--tune` to search for hyperparameters first (`RandomizedSearchCV`, stratified CV) and write the
best ones back into the experiment's `model_params` before training on the full training set:

```powershell
python scripts/train.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml --tune
```

Trained pipelines (preprocessing + model) are saved to `outputs/models/<experiment_name>.joblib`,
and metrics to `outputs/metrics/<experiment_name>.json`.

Both datasets are severely class-imbalanced (UNSW-NB15 is 95.6% `Benign`, CIC-IDS2017 84.9%). **All four
models handle this by reweighting the training objective with the same per-class weight
`w_c = n / (k * count_c)` (scikit-learn's `class_weight="balanced"` formula), computed on the training
split only. None of them resample rows.**

The weight *formula* is identical across models -
`sklearn.utils.class_weight.compute_sample_weight("balanced", y)` is literally the per-sample
broadcast of `class_weight="balanced"`, and both equal `n / (k * count_c)` (verified numerically; the
weights average to 1). The *values* are identical for `random_forest`, `xgboost` and `neural_network`,
which all fit the full training split. `svm`'s are computed on its 200k final-fit subsample, whose
500-row class floor makes the rare classes less rare, so its weights for them are smaller (UNSW-NB15:
`Worms` 168 instead of 1,182, `Shellcode`/`Backdoor`/`Analysis` 40 instead of 100-139). What differs is only how each objective consumes them: weighted impurity/leaf
counts at each split (`random_forest`), per-class scaling of the penalty `C` (`svm`), per-sample
gradient/Hessian scaling (`xgboost`), per-sample cross-entropy scaling (`neural_network`). So they are
equivalent in principle and in the weights, not bit-identical in optimisation behaviour.

- `random_forest`, `svm`: `class_weight="balanced"` constructor argument.
- `xgboost`: `XGBClassifier` has no `class_weight` argument at all, only per-sample weights at fit time,
  so `pipeline.py`'s `_MODELS_REQUIRING_SAMPLE_WEIGHT` passes
  `compute_sample_weight("balanced", y)` - the same weights, one per row - through `fit()`.
- `neural_network`: `TorchMLPClassifier` (`src/xai_thesis/torch_mlp.py`), a small PyTorch MLP
  (`Linear → ReLU` per hidden layer → `Linear` to raw logits) that passes those same per-class weights
  to `nn.CrossEntropyLoss`. It replaced scikit-learn's `MLPClassifier`, which has no
  `class_weight`/`sample_weight` support and had therefore forced a bespoke
  resample-to-the-median-class-count path (`balance_classes`); that path and its
  `smote_balance_classes` variant are both gone. The wrapper keeps sklearn semantics
  (`fit`/`predict`/`predict_proba` with string labels, cloneable for `RandomizedSearchCV`), seeds
  torch/NumPy/the DataLoader generator and enables `torch.use_deterministic_algorithms` so SHAP/LIME
  results are reproducible, carries `max_iter` over as the epoch count and `learning_rate_init`/`alpha`
  as the Adam LR / weight decay, and does stratified-validation early stopping when the config sets
  `early_stopping: true`.

Because reweighting never changes which rows exist, each `RandomizedSearchCV` fold now trains on its
real rows and computes its own fold weights - the fold-leak caveat the old resampling path carried is
gone.

`XGBClassifier` also
rejects this project's raw string class labels outright (verified directly: fitting on
`"Benign"`/`"Analysis"`/etc. raises `ValueError`, unlike every other model here, which accepts
them natively) - `models.py`'s `_StringLabelXGBClassifier` subclasses it to label-encode
internally, so `.classes_`/`.predict()` still return the original string labels like every other
model's contract requires (`evaluate_classifier`, `compute_shap_values`, and the class-name lookups
in `pipeline.py` all assume this). It's a real subclass, not a wrapper, specifically so
`shap.TreeExplainer` still recognizes it directly (verified).

`random_forest` was originally dropped from the model catalog (shared `gradient_boosting`'s
tree-ensemble family and `TreeExplainer`, consistently showed the highest cross-model SHAP agreement
with it of any model pair, and underperformed it on 2 of 3 datasets - `gradient_boosting` has since
been removed from the catalog entirely, see the blockquote above) - `random_forest` was re-added by
explicit request despite that redundancy concern still applying. It uses `class_weight="balanced"`
and `TreeExplainer`, same as `xgboost`.

An earlier `svm` (`LinearSVC`) was dropped for substantially worse macro-F1 than the other models,
replaced at the time by `logistic_regression` as the catalog's linear model (itself later removed
entirely, see the blockquote above). The current `svm` entry is a
different, later addition: `SVC` with a **linear kernel** and `probability=True`, matching an external
reference configuration exactly for those two settings. `probability=True` is kept despite the extra
internal Platt-scaling cost it adds on top of the kernel fit, to match that reference configuration
directly; scikit-learn 1.9 deprecates this in favor of wrapping with `CalibratedClassifierCV`, but the
parameter still works.

`svm` is measurably more expensive than every other model in this catalog, at both training and
explanation time: an untuned fit took ~26 minutes on UNSW-NB15's old 10%-sample (~101k training
rows) - slower than an earlier RBF-kernel attempt without `probability=True` (~5.5 minutes on
UNSW-NB15), confirming the calibration cost, not the kernel choice, dominates. Explanation generation is
slower too: `svm` has no structural SHAP explainer (same permutation-based fallback as `neural_network`),
and each permutation sample pays `SVC`'s comparatively expensive prediction cost, pushing SHAP generation
alone to 20-30 minutes for UNSW-NB15's earlier binary version, and ~105 minutes for its later
10-class multiclass version (permutation SHAP cost scales with class count, not just row count).

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
  `200_000` default) so the search phase stays tractable. (The final fit is subsampled too, but separately - see
  `_FINAL_FIT_SUBSAMPLE` below.) With 8 candidate `C` values and `n_iter=8`, the search covers the
  whole grid.
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
macro-F1 from 0.313 to **0.360**.
The final fit still hit the `max_iter` cap without fully converging (visible via a
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
data (confirmed a genuine deadlock via live CPU-time sampling - zero CPU usage across repeated
5-second samples - not slowness); fixed by calling `tune_hyperparameters()` directly with an
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

**CIC-IDS2017's `svm` is now trained, as of 2026-08-17.** `configs/experiments/cic_ids2017_svm_baseline.yaml`'s
`C` search completed early on (`C=10`, ~8 minutes thanks to an explicit `subsample_size=40_000` override
well below the shared 200k default - CIC-IDS2017 is ~2M rows), but the final fit on the complete,
un-subsampled training set repeatedly failed to *finish* across several separate attempts, for unrelated
reasons each time: one ran 24+ hours unattended (confirmed still actively computing throughout via rising
CPU time, not hung - `max_iter` bounds iteration *count*, but at this row count each iteration itself
costs far more than on any other dataset here, so the cap doesn't translate to a comparable wall-clock
bound) before an unrelated session restart interrupted it with no checkpoint to resume from; a later
attempt was intentionally cancelled once CIC-IDS2017's feature set was about to change again (see
Datasets above) and would have made the result stale immediately; a third didn't survive an actual
machine shutdown over a multi-day gap. The same `_FINAL_FIT_SUBSAMPLE` fix that resolved UNSW-NB15's
`svm` above (stratified 150,000-row subsample for the final fit, `unsw_nb15`/`cic_ids2017` specifically)
already applied here the whole time - it just needed one uninterrupted run to prove it out. That run
completed cleanly: macro-F1 **0.518**, accuracy 0.855, roc_auc_macro 0.986 (~25 minutes for the final
fit, plus ~76 minutes for permutation-based SHAP generation - the slow part of this model's explanation
suite, not its training).

**`_FINAL_FIT_SUBSAMPLE` was removed entirely on 2026-08-27** (explicit user request): `svm`'s final fit
on `unsw_nb15`/`cic_ids2017` was made to train on the complete training set, same as every other model in
this catalog, rather than the 150,000-row stratified subsample described in the two paragraphs above - a
deliberate reversal of that deviation so `svm`'s reported metrics would be trained the same way as every
other model's and stay directly comparable, accepting the reintroduced risk of a repeat of CIC-IDS2017's
24+-hour non-completion history.

**That reversal was itself reverted the next day, 2026-08-28, after `unsw_nb15`'s `svm` final fit failed
to complete twice in a row on the (also newly-corrected) full raw data** - once uncapped (18+ hours,
confirmed still actively computing via CPU-time sampling, not hung), once capped at `max_iter=5000` (a
lower iteration ceiling intended to bound wall-clock time), which still ran 4h8m without finishing. The
conclusion: at this row count, each individual libsvm iteration is itself the expensive part, so capping
iteration count alone isn't a reliable lever - `_FINAL_FIT_SUBSAMPLE` is back, and this time it also
covers `cic_ids2017`, at explicit user request, even though that dataset's `svm` had already trained
successfully on its complete set - for methodological consistency across both datasets rather than
"whichever finished in time keeps the full set."

**The subsample was raised again the same day, 150,000/`min_per_class=200` -> 200,000/`min_per_class=500`**,
after the first subsampled retrain put `unsw_nb15`'s `svm` (macro-F1 0.413) behind `neural_network`
(0.449). Checked before changing anything, not assumed: raising `target_size` alone adds rows almost
entirely to already-well-represented classes (`unsw_nb15`'s `Benign` absorbs ~90% of a 150k->200k
increase), since every class sitting at the `min_per_class` floor gets zero extra rows regardless of
`target_size` - not a lever likely to move macro-F1, which weights every class equally including the
weak ones. `min_per_class` is what actually matters for the floored classes: raising it 200->500 gives
`unsw_nb15`'s `Analysis`/`Backdoor`/`Shellcode` and `cic_ids2017`'s `Web Attack`/`PortScan`/`Bot` a full
500 rows each (all six have well over 500 available). `unsw_nb15`'s `Worms` (118 total) and
`cic_ids2017`'s `Infiltration` (24 total) are unaffected by either parameter - both floors already
exceed their entire natural supply.

**A real config bug surfaced during this retrain, not just a data/methodology question**: both
`unsw_nb15_svm_baseline.yaml` and `cic_ids2017_svm_baseline.yaml` had picked up a `max_iter=5000`
override during the earlier convergence investigation above; when that approach was abandoned in favor
of the subsample, only `unsw_nb15`'s override was cleaned up. `cic_ids2017`'s stale `max_iter=5000`
survived into the 200k/500 retrain and, on a now-larger and more imbalanced-in-absolute-terms subsample,
produced a badly degenerate fit (accuracy 0.182, macro-F1 0.135 - confirmed via `ConvergenceWarning` in
the run log, not a real finding about the larger subsample). Fixed and retrained again with the correct
default `max_iter=50_000` cap.

**Final results — stratified random 70/30 split, `random_forest`, `neural_network` and `svm` re-tuned
(`RandomizedSearchCV`, `f1_macro`) and all four retrained, `svm` final fit on the 200k /
`min_per_class=500` subsample. `xgboost` keeps an earlier search's params, chosen by test score - see
"Hyperparameter provenance" below. Regenerated 2026-08-31/09-01.**

| Model | UNSW-NB15 macro-F1 | UNSW-NB15 accuracy | CIC-IDS2017 macro-F1 | CIC-IDS2017 accuracy |
| --- | --- | --- | --- | --- |
| `random_forest` | **0.609** | 0.977 | 0.888 | 0.999 |
| `xgboost` | 0.597 | 0.976 | **0.901** | 0.999 |
| `neural_network` | 0.463 | 0.971 | 0.428 | 0.942 |
| `svm` | 0.419 | 0.968 | 0.379 | 0.864 |

Full per-metric numbers are in `outputs/metrics/*_baseline.json`; `roc_auc_macro` is 0.97-0.9996
for every model/dataset, so `f1_macro` (which weights the rare classes equally) is the
discriminating headline metric.

- **`xgboost` is best on CIC-IDS2017 (0.901), `random_forest` best on UNSW-NB15 (0.609)** - the two
  tree ensembles are within ~0.01-0.02 of each other on both datasets and clear of `neural_network`
  and `svm`.
- **Re-tuning on the new split moved things.** `random_forest` on CIC-IDS2017 jumped from 0.720
  (old params reused) to 0.888 once the search dropped `max_features="log2"` for `0.5` - the same
  column-count sensitivity documented for the earlier feature-set cut. `neural_network` went from a
  collapsed 0.22/0.30 (old `MLPClassifier`-era learning rates, unstable for the PyTorch MLP) to
  0.43/0.46 once `learning_rate_init` was searched over a lower range.

#### Hyperparameter provenance

- **Search settings as reported in the thesis:** `tune_hyperparameters()` - `RandomizedSearchCV`,
  30 sampled configurations (`svm`: its 8 `C` values, i.e. the whole grid), stratified 3-fold CV,
  `f1_macro`, on a class-proportional subsample of 200,000 training rows (the function's default,
  and the only value `scripts/train.py --tune` uses).
- **The stored params are not exactly reproducible.** No record of the 2026-08-31/09-01 search runs
  survives. Re-running the `unsw_nb15` `random_forest` search on 2026-10-04 (same seed, data and
  grid) gave different params at 200,000 rows (`n_estimators=200`, `max_features="sqrt"`) than the
  config holds (`100`, `0.5`); at 40,000 rows it reproduced the *pre*-re-tune config exactly. The
  same held for the code as of the morning of 2026-08-31. So the configs come from search runs
  whose exact settings cannot be reconstructed.
- **`xgboost` was selected by test-set score (commit `864b376`, 2026-09-01).** Both `xgboost` configs
  were reverted from the stratified-split re-tune to an earlier search result - run on the previous,
  chronological split - because the earlier params scored higher on the **test** set. Test-set
  selection biases the reported `xgboost` scores upward. Cross-validating both candidates on the
  training partition (3-fold, 200,000-row class-proportional subsample, seed 42; 2026-10-04) prefers
  the re-tuned params on both datasets:

  | dataset | test `f1_macro` kept / re-tuned | training-CV `f1_macro` kept / re-tuned |
  | --- | --- | --- |
  | UNSW-NB15 | 0.597 / 0.554 | 0.557 / 0.559 |
  | CIC-IDS2017 | 0.901 / 0.893 | 0.798 / 0.802 |

  The models were kept anyway, because the expert interviews had already been run on the
  CIC-IDS2017 `xgboost` explanations. The model ranking is the same under either config: `xgboost`
  is best on CIC-IDS2017 (0.893 or 0.901 vs `random_forest` 0.888), and second on UNSW-NB15 behind
  `random_forest` (0.609), ahead of `neural_network` (0.463). The thesis discloses the selection.
- **`svm` is still the weakest on both** and essentially unaffected by re-tuning (`C` moved 10->3
  on UNSW-NB15 for no real change).
- The stratified split is genuinely easier than the old chronological one - train and test are now
  i.i.d. rather than temporally separated - so every model's macro-F1 rose relative to the
  chronological-split numbers, most sharply for the tree ensembles on CIC-IDS2017 (`xgboost`
  0.79->0.89, `random_forest` 0.63->0.89).

**`xgboost` added 2026-08-24, in scope for the thesis (see the blockquote above).** Tuned
(`RandomizedSearchCV` on a reduced search subsample for tractability, same pattern used for every
model) and trained on the complete, un-subsampled training set for both - no
`_FINAL_FIT_SUBSAMPLE` exception needed, unlike `svm`: `tree_method="hist"` finishes CIC-IDS2017's
final fit in a couple of minutes regardless of dataset size. See the results table above for current
numbers.

## Analysis

Confusion matrix (absolute counts, colored on a symmetric-log scale so cells of very different
magnitude all stay visually distinguishable under severe class imbalance; the color scale's top end is
anchored to the largest value *excluding* the single most extreme cell, so one dominant majority-class
cell doesn't compress every other cell's shade into the same dark band) for a trained experiment, or all
experiments in `configs/experiments/`:

```powershell
python scripts/confusion_matrix.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml
python scripts/confusion_matrix.py  # all experiments
```

Plots are written to `outputs/plots/<dataset_name>/<model>_baseline/performance/confusion_matrix.png`,
with the counts next to them in `confusion_matrix.csv` (rows true class, columns predicted class, Benign
first) - the source for every confusion count quoted in the thesis.

Aggregate one-vs-rest ROC curves - macro- and micro-averaged, plus the chance diagonal, with each AUC in
the legend - for a trained experiment, or all of them:

```powershell
python scripts/roc_curve.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml
python scripts/roc_curve.py  # all experiments
```

Written to `outputs/plots/<dataset_name>/<model>_baseline/performance/roc_curves.png`. Deliberately aggregated (no
per-class curves - unreadable at 7-10 classes, and that detail is already in the confusion matrix and
per-class F1): the macro-average AUC equals the `roc_auc_macro` in `outputs/metrics/` and is the one to
read; the micro-average is `Benign`-dominated and hugs the top-left corner regardless of rare-class
separation - the two are shown together to make that divergence visible.

Label class distribution for a dataset, or all datasets:

```powershell
python scripts/label_distribution.py --dataset unsw_nb15
python scripts/label_distribution.py  # all datasets
```

Plots are written to `outputs/plots/<dataset_name>/data/label_distribution.png` (processed train+test,
i.e. after corrupted-row removal and deduplication) and `label_distribution_raw.png` (every raw record,
after label normalization and consolidation, nothing removed - the datasets chapter's figures), with the
raw counts in `outputs/metrics/dataset_inspection/<dataset_name>/label_distribution_raw.csv`. `outputs/plots/`
is organized per dataset first, then grouped by purpose (since 2026-09-14), so no plot sits loose in
a dataset or model folder:

```text
outputs/plots/<dataset_name>/
├── data/                   label_distribution(_raw), missing_values, train_test_split, feature_correlation
├── agreement/              local_agreement_{feature,rank} (SHAP vs LIME)
└── <model>_baseline/
    ├── performance/        confusion_matrix (.png + .csv), roc_curves
    ├── global/             shap_bar, permutation_importance
    └── interview/<class>/  shap_waterfall_{correct,misclassified}, lime_{correct,misclassified}
```

The folder helpers are `_dataset_plot_dir` and `_experiment_plot_dir` in `pipeline.py`.

Missing-value summary, train/test split balance, and feature correlation, for a dataset or all
datasets:

```powershell
python scripts/dataset_diagnostics.py --dataset unsw_nb15
python scripts/dataset_diagnostics.py  # all datasets
```

Plots are written alongside `label_distribution.png` in the same `data/` folder:
`outputs/plots/<dataset_name>/data/missing_values.png`, `train_test_split.png`, and
`feature_correlation.png`.

## Explainability

Three techniques are generated per trained experiment (or all of them) via:

> **Partial dependence plots (PDP) were removed entirely on 2026-08-27** (had been out of the
> thesis's explainability scope since 2026-07-31, and out of `generate_explanations()`
> specifically since 2026-08-17, but kept as a still-callable function until now).
> `plot_partial_dependence` no longer exists in `explainability.py` - not just unused, deleted -
> along with its tests and every `partial_dependence.png` file that existed anywhere in
> `outputs/`.

```powershell
python scripts/explain.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml
python scripts/explain.py  # all experiments
```

- **SHAP**: `TreeExplainer` for `random_forest` and `xgboost` (exact, deterministic), and a
  permutation-based `Explainer` for `neural_network` and `svm` (no fast structural explainer exists for
  a plain MLP or a kernel SVM - this path is markedly slower for both, and slower still for `svm`
  specifically, since its own predictions are more expensive to compute). The permutation path is
  **seeded** (`random_state`, default 42, passed through to `shap.Explainer(seed=...)` and the
  `shap.sample` background sub-draw) so `svm`/`neural_network` SHAP values are reproducible run to run,
  and runs at a **fixed permutation count** rather than SHAP's default: `max_evals =
  _SHAP_PERMUTATION_MULTIPLIER * (2 * n_features + 1)` with the multiplier at 50, i.e. 50 permutations
  per instance instead of the ~4 that SHAP's `max_evals=500` default works out to at this project's
  52-59 post-one-hot feature count. At 4 permutations the estimate was visibly unstable (per-instance
  top-8 feature sets moved by ~10-25% and their ordering by more between independent runs); 50 settles
  it. The multiplier is the knob to lower if the `svm` permutation path becomes intractable (though
  for CIC-IDS2017's global sample a chunked parallel run was used instead, to keep it equal across
  models - see Global explanations) - cost
  scales ~linearly in it (CIC-IDS2017 `svm`: ~48 s/instance at 4 permutations, ~430 s at 50).
  `generate_explanations` produces a global bar plot (mean |SHAP value| per feature, stacked by
  class). Local waterfall plots are produced only as interview stimuli (see Interview stimuli below).
- **LIME** (`LimeTabularExplainer`): local explanations for the same interview-stimulus instances as
  the SHAP waterfalls, so SHAP and LIME can be compared directly on identical cases. Rendered via `explain_instance_with_lime` in
  `explainability.py`, LIME's own unmodified `explanation.as_pyplot_figure()` - the single-panel
  horizontal bar chart LIME ships by default, not a custom reproduction of anything. A custom three-panel
  reproduction of LIME's `show_in_notebook()` layout (`explain_instance_with_lime_full`) existed briefly
  and was used by the pipeline for a while, but was removed entirely on 2026-08-28 at explicit user
  request, once it became clear this project should only ship each library's own native, standardized
  plot types rather than custom reconstructions of a view that library doesn't itself export as a static
  image. LIME's real native plotting surface is just this one function -
  `show_in_notebook`/`save_to_file` only produce an HTML+D3 widget with no static-image path, and there
  is no other LIME-native alternative to fall back on.
  Continuous features are discretized into training quartiles on **both** datasets (LIME's default
  `discretize_continuous=True`), which is what gives the bars their range labels (`dttl > 29.00`)
  rather than bare feature names. CIC-IDS2017 used to run without it, because the discretizer
  crashed there (scipy `truncnorm`: "scale must be positive"). That was blamed on column skew, but
  the real cause was the 1,120 NaN cells in `Flow Bytes/s` and `Flow Packets/s`, which make LIME's
  per-bin standard deviation NaN. `_build_lime_explainer` now fills numeric NaNs, in the training
  data and in each explained row, with `x_train`'s column medians. Those are the same values the
  pipeline's own `SimpleImputer(strategy="median")` uses, so model predictions are unchanged.
- **Permutation feature importance** (`sklearn.inspection.permutation_importance`): run on the full
  pipeline, so importances are reported in terms of the original feature names, not the one-hot-expanded
  ones. Each feature's importance is its macro-F1 drop (10 shuffles, mean with standard-deviation error
  bars), drawn as a horizontal bar **stacked by class**, like the SHAP bar: macro-F1 is the mean of the
  per-class F1 scores, so the drop splits exactly into each class's (1/K-scaled) F1 drop, and the
  segments sum to the bar. A class's segment can be negative (shuffling slightly helped that class) and
  is then drawn left of zero rather than clipped. Each class keeps the same colour in both global plots
  (`class_colors`), so they can be read side by side. The chart is hand-built, unlike the SHAP/LIME
  plots above - checked directly against `sklearn.inspection`'s actual contents (2026-08-28) and
  confirmed there is no native display class for this technique the way `PartialDependenceDisplay`
  exists for PDP, so a custom chart is the established way to visualize this, not a deviation from one.

`generate_explanations()` used to also write a global SHAP beeswarm for the class each model handles
worst (`_hardest_class_and_confusion`), plus local SHAP waterfall + LIME plots for one correct and one
misclassified instance of that class. All three were removed on 2026-09-14 because the thesis uses
none of them: local plots are now produced only as interview stimuli (below), and the old
`shap_beeswarm.png`, `lime_*.png` and `shap_waterfall_*.png` files were deleted from `outputs/plots/`.

Plots are written to `outputs/plots/<dataset_name>/<model>_baseline/global/`.

### Limitation: SHAP base-value footing across models

Class balancing is no longer a source of cross-model inconsistency - all four models reweight the
objective with the same `n / (k * count_c)` formula, computed on the rows each is fitted on (`svm` on
its 200k final-fit subsample, the others on the full training set). Two explainer-mechanics points remain, worth
stating when writing this up:

- **SHAP base values are not all on one footing.** `TreeExplainer` runs
  `feature_perturbation="tree_path_dependent"` with no background (xgboost's SHAP build rejects the
  interventional+background path), so `random_forest`/`xgboost` base values come from each model's
  own weight-adjusted internal prior - near `1/n_classes` for `random_forest`, margin space for
  `xgboost`. The permutation path (`svm`, `neural_network`) uses an explicit background; `pipeline.py`
  draws that background **class-balanced** (`_class_balanced_sample`, ~100 rows, ~even per class)
  rather than a uniform random sample that would be 85-96% `Benign`, so `svm`/`neural_network` land on
  roughly the same near-uniform footing as `random_forest`. `xgboost`'s base value remains margin-space
  and comparable in attribution *direction* only, not level.
- **The global sample is class-balanced, not prevalence-weighted.** The global SHAP bar and
  permutation importance explain a **500-row class-balanced** test sample (`_class_balanced_sample`,
  `random_state=42`) - 50 rows/class on UNSW-NB15 (all 10 classes), 71/class on CIC-IDS2017 except
  `Infiltration`, which has only 11 test rows and contributes all of them (437 rows total). This
  replaced a 200-row *stratified* draw that was 191/200 `Benign` on UNSW-NB15 and left 7 of 10 classes
  with zero rows, making the global rankings a two-class problem. The trade-off: the rankings no longer
  reflect deployment prevalence (documented separately in `label_distribution.png`), and the
  permutation-importance macro-F1 drop is measured under the rebalanced distribution - intentional, and
  noted in `generate_explanations()`. Global SHAP and PFI use this identical sample.

### Interview stimuli

The only source of local SHAP/LIME plots (`generate_interview_stimuli` in `pipeline.py`): concrete
examples to show expert-interview subjects. It picks classes a security
practitioner would recognize by name, so interview time isn't spent explaining an unfamiliar
label: `Reconnaissance`/`Exploits` for UNSW-NB15, `DoS/DDoS`/`Brute Force`/`Bot` for CIC-IDS2017
(`_INTERVIEW_STIMULUS_CLASSES` in `pipeline.py` — see its comment for the excluded classes, and for
why `PortScan` was dropped on 2026-09-01: removing the leaky `Destination Port` column collapsed it
under dedup from ~159k rows to ~2k, so it no longer samples real scanning traffic). For each
configured class the model's most common real misclassification target is derived dynamically from
the test set (`_most_common_confusion`) rather than hand-picked, then one correct and one
misclassified SHAP waterfall + LIME plot are generated (`_generate_instance_explanations`), under
`outputs/plots/<dataset>/<model>_baseline/interview/<class>/`.

```powershell
python scripts/interview_stimuli.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml
python scripts/interview_stimuli.py  # all experiments
```

Plots are written to `outputs/plots/<dataset_name>/<model>_baseline/interview/<class>/` (e.g.
`outputs/plots/unsw_nb15/random_forest_baseline/interview/Reconnaissance/`). A class maps to no
plots at all if the model never misclassifies it in the test set (no stimulus pair possible); every
class in the current list was re-checked to have both a correct and a misclassified test instance
for all four models.

### Interview stimulus candidates

`scripts/stimulus_candidates.py` (`src/xai_thesis/stimuli.py`) profiles each model's errors on what
makes a misclassification a good interview stimulus - one the analyst has to reason about rather
than dismiss. The model is **confident** in its wrong answer (`confidence` = the probability it gave
the class `.predict()` returned, which is the class a waterfall stimulus explains), and the flow
carried **real traffic in both directions**.

```powershell
python scripts/stimulus_candidates.py                  # the agreement instance set
python scripts/stimulus_candidates.py --scope benign   # every benign test flow
```

- `--scope instance_set` (default) profiles the class-balanced set the local agreement panel uses,
  so SHAP and LIME attributions already exist for every row. It holds only 6 (UNSW-NB15) / 8
  (CIC-IDS2017) benign rows, so it **cannot** say whether confident false positives exist.
- `--scope benign` profiles every benign test flow and keeps the false positives, which is how that
  question is actually answered.

Errors are typed IDS-style, with benign as the negative class: false positive (a benign flow the
model alerted on), false negative (an attack predicted benign), attack confusion (an attack put in
the wrong attack class). Responder traffic is `none` / `packets_only` / `payload`, but `payload`
only means "more than zero bytes", which a 6-byte exchange passes - so a candidate additionally
needs `--min-bytes` (default 100) in **each** direction. Output goes to
`outputs/metrics/stimuli/<dataset>_error_profile.csv` and `<dataset>_benign_false_positives.csv`;
each model's rows are replaced independently, so a slow model can be rerun on its own.

### Global explanations (descriptive)

`generate_explanations()` writes, per model and dataset, `shap_bar.png` and
`permutation_importance.png`, both on the **same** class-balanced test sample
(`_class_balanced_sample`, `total=500`, `random_state=42`: 500 rows on UNSW-NB15, 50/class; 437 on
CIC-IDS2017, 71/class with `Infiltration` contributing all 11 of its test rows). The only comparison
between the two global rankings is the number of features their top 10s share (feature agreement at
`k = 10`, `global_overlap_table` in `explanation_report.py`, written to
`outputs/metrics/explanation_report/global_overlap.csv`); nothing is compared across models. (A
SHAP-vs-PFI agreement panel over several k and a cross-model overlap table existed until 2026-10-01;
both were removed with their outputs.)

- **Global SHAP importance** = mean `|SHAP|` over instances *and* classes. `shap_bar.png` stacks the
  per-class means, so a bar's total length is `n_classes` times this value; the ranking is the same.
- **Permutation feature importance** = mean macro-F1 drop over `n_repeats=10` shuffles, on the full
  pipeline (raw features), same sample.

Both vectors are cached per experiment as `outputs/metrics/global_importance/<experiment>__shap.npz`
and `__pfi.npz` (raw column order, plus the sample's `instance_index` and class counts), which the
explanation report reads.

**CIC-IDS2017 `svm` global SHAP is reported, but not recomputed by default.** Its result comes from
one chunked run (see the note below) and is part of the thesis results like every other model's. A
rerun skips it, because its permutation SHAP takes ~440 s/instance,
~53 h for the 437-row sample in one call. `generate_explanations` skips global SHAP when the
experiment config's `skip_global_shap` flag is true. The flag defaults to true for that pair only
(`_GLOBAL_SHAP_SKIPPED_BY_DEFAULT`), and `cic_ids2017_svm_baseline.yaml` also sets it explicitly. The
skip is logged as a warning, and PFI is still computed. Setting the flag to false and passing
`--shap-chunk-size`/`--shap-n-jobs` runs the SHAP as checkpointed parallel chunks
(`_compute_shap_in_chunks`; chunks are kept under `outputs/metrics/global_importance/chunks/`).

> That chunked run **was done once, on 2026-10-01** (88 chunks of 5 rows, 8 at a time, 7.7 h), so a
> CIC-IDS2017 `svm` global SHAP result exists in the caches, `shap_bar.png` and `global_top10.csv`.
> The method is the same (50 permutations, same background, same 437 rows), but values are not
> bit-identical to a single call: shap 0.52.0 seeds NumPy's global generator once per explainer, and
> chunk `c` is seeded `random_state + c`. The skip flag stops a rerun from repeating it. It does not
> delete the result.

Why the global sample is class-balanced: a 200-row *stratified* draw (the basis before 2026-09-09) is
191/200 `Benign` on UNSW-NB15 and contains only 6 of UNSW-NB15's 10 classes and 3 of CIC-IDS2017's 7,
so its "global" ranking describes a handful of classes. Doubling the balanced sample to ~1000 rows
(`random_forest`, both datasets) left the top-10 SHAP feature set unchanged, with one adjacent swap
each, so 500 rows is enough.

### Local disagreement analysis: SHAP vs LIME

The project's **only** agreement analysis. It uses two top-k metrics from **Krishna et al., "The
Disagreement Problem in Explainable Machine Learning" (TMLR 2024)**. `T_k` is the set of the `k`
features with the largest `|attribution|`:

- **Feature agreement** `FA_k = |T_k(SHAP) n T_k(LIME)| / k`.
- **Rank agreement** `RA_k = |{f in T_k(SHAP) n T_k(LIME) : rank_SHAP(f) == rank_LIME(f)}| / k`,
  where a rank is the feature's 1-based position in the `|attribution|` ordering. `RA_k <= FA_k`, and
  `agreement_fraction` asserts it.

**Ties** in `|attribution|` are broken deterministically by **feature name** (ascending string order;
`agreement.ranked_feature_indices`), so every `T_k` and every rank is unique and reproducible. This
replaced the earlier port of the authors' reference code, which left ties to `np.argsort` and gave
tied features a shared dense rank. On the cached attribution matrices no two features tie anywhere
in a top 21, and recomputing all published numbers with the new rule changed nothing beyond
4-decimal rounding.

```powershell
python scripts/agreement.py --dataset unsw_nb15
python scripts/agreement.py                               # every dataset
python scripts/agreement.py --overwrite                   # recompute the attribution caches
```

- **Sample**: `_class_balanced_sample(total=60, random_state=42)` of the test split: **60** rows on
  UNSW-NB15 (6/class x 10) and **56** on CIC-IDS2017 (8/class x 7). Rows are included whether or not
  the model classifies them correctly. The draw depends only on the test split and the seed, so
  **all four models of a dataset are scored on the identical instances**. Every config uses
  `random_state: 42`, `generate_agreement_heatmaps` raises if seeds differ, and it raises if any two
  models' cached instance sets differ.
- **Explained class**: both methods explain the class the model predicts, the `predict_proba` argmax
  (LIME's `top_labels=1`). SHAP is taken for that same class.
- **Feature space**: original features. SHAP's one-hot `proto`/`service`/`state` columns are summed
  back into their parent feature (`compute_shap_values`), which is the space LIME already works in (40
  features for UNSW-NB15, 59 for CIC-IDS2017).
- **k** in `{1, 5, 10}`. **`k = 10` is the headline value.**
- **Aggregation**: per model and dataset, mean +/- standard error across instances, where SE is the
  sample standard deviation (`ddof=1`) divided by `sqrt(n)`. `n` is reported alongside.

Outputs: `outputs/plots/<dataset>/agreement/local_agreement_{feature,rank}.png` (models x k, fixed
`[0, 1]` sequential `Reds` scale) and `outputs/metrics/agreement/<dataset>_local_agreement.csv`
(`dataset, model, k, fa_mean, fa_se, ra_mean, ra_se, n`). The attribution matrices are cached as
`outputs/metrics/agreement/attributions/<experiment>.npz`, because `svm`'s permutation SHAP is
~90 s/instance on UNSW-NB15 and ~410 s on CIC-IDS2017.

**Local agreement at k = 10** (mean +/- SE; n = 60 UNSW-NB15, 56 CIC-IDS2017):

| model | UNSW-NB15 FA@10 | UNSW-NB15 RA@10 | CIC-IDS2017 FA@10 | CIC-IDS2017 RA@10 |
| --- | --- | --- | --- | --- |
| `random_forest` | 0.375 +/- 0.018 | 0.045 +/- 0.007 | 0.389 +/- 0.015 | 0.041 +/- 0.009 |
| `xgboost` | 0.357 +/- 0.016 | 0.075 +/- 0.010 | 0.282 +/- 0.016 | 0.030 +/- 0.007 |
| `neural_network` | 0.623 +/- 0.017 | 0.090 +/- 0.010 | 0.434 +/- 0.023 | 0.057 +/- 0.009 |
| `svm` | 0.692 +/- 0.010 | 0.172 +/- 0.016 | 0.545 +/- 0.014 | 0.107 +/- 0.014 |

> **`svm` caveat.** `SVC.predict()` (one-vs-one voting) and `predict_proba`'s argmax disagree on 34
> of the 60 sampled UNSW-NB15 instances, a known `SVC` + Platt-scaling effect under class imbalance
> (0 disagreements for every other model). The agreement numbers stay valid, since SHAP and LIME both
> explain the `predict_proba`-argmax class, but for those instances that class is not the one
> `.predict()` returns.

`_AGREEMENT_EXCLUDED` (pairs left out of the local panel) is empty, and a test pins it. Lowering
`_SHAP_PERMUTATION_MULTIPLIER` for `svm` alone was rejected: an inconsistent permutation count across
models would make the cross-model reading of the local panel meaningless.

### Explanation report

`scripts/explanation_report.py` (`src/xai_thesis/explanation_report.py`) writes the thesis tables to
`outputs/metrics/explanation_report/`. It only reads caches (it never computes SHAP, LIME or PFI), and
it redraws both samples to check that every cache was computed on exactly those rows.

```powershell
python scripts/explanation_report.py
```

| file | contents |
| --- | --- |
| `local_agreement.csv` | **thesis table**: `dataset, model, k, fa_mean, fa_se, ra_mean, ra_se, n` for k = 1, 5, 10 |
| `global_top10.csv` | **thesis table**: top-10 features per dataset, model and method (`SHAP`, `PFI`), with `rank` and `importance` |
| `global_overlap.csv` | **thesis table**: `dataset, model, k, n_shared, feature_agreement, shared` - SHAP vs PFI top-10 overlap |
| `sample_sizes.csv` | per-class row counts of the local and the global sample, per dataset (also logged) |
| `<dataset>_local_instances.csv` | per instance: true, predicted and explained class, FA/RA (and supplementary sign agreement) per k |
| `<dataset>_local_top10.csv` | signed SHAP and LIME top 10 per instance, with direction (toward/against the explained class) |
| `<dataset>_local_disagreement.csv` | per instance: top-10 membership disagreements and sign disagreements |
| `<dataset>_local_summary.csv` | per model, over all instances and broken down into correct / misclassified |

- **Global ranks** order by importance, descending, with ties broken by feature name. A negative PFI
  (shuffling slightly *improved* macro-F1) means no measurable importance, so it ranks below zero
  rather than by its magnitude. Local ranks use the `|attribution|` ordering of the metrics.
- **Sign** (local only): both methods explain the `predict_proba` argmax class, so positive means
  "toward that class". LIME is discretized, so its sign refers to the instance's quartile bin, while
  SHAP's is relative to the background sample. Sign agreement (Krishna et al.) is a supplementary
  per-instance column, not one of the two thesis metrics.
- **`correct`** compares the true class to `.predict()`. For `svm` that can differ from the explained
  argmax class (`predict_disagrees_with_proba`; see the `svm` caveat above).
