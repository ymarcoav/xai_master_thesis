# xai_master_thesis

Code for the master thesis titled "Evaluating Explainable AI Methods for Intrusion Detection Systems: From Model Explanations to Security Decisions".

## Structure

- `configs/datasets/`: one YAML per **active** dataset (raw/processed paths, label column, columns to drop, categorical columns).
- `configs/experiments/`: one YAML per **active** trained model (dataset, model type, hyperparameters, metrics to report).
- `configs/_archive/`: dataset/experiment configs for out-of-scope datasets (NSL-KDD, DAPT2020) - archived 2026-08-17, still fully functional, just not touched by the default (no-args) pipeline commands. See the Datasets/Training sections below. (`gradient_boosting`/`logistic_regression` were archived here too for a while, then removed from the project entirely on 2026-08-24 - see Training below.)
- `data/`: raw and processed datasets. Data files are ignored by Git.
- `data/_archive/`: raw/processed data for archived datasets (also gitignored).
- `src/xai_thesis/`: reusable project code (dataset adapters, preprocessing, models, evaluation, explainability, pipeline orchestration).
- `scripts/`: command-line entry points for inspection, preprocessing, training, and analysis.
- `notebooks/`: exploratory analysis and thesis figures in progress.
- `outputs/`: generated models (`models/`), metrics (`metrics/`), and plots (`plots/`).
- `outputs/_archive/`: models/metrics/plots for archived datasets/models. Also gitignored.
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
| `scikit-learn` | 1.9.0 | `permutation_importance` (global panel, PFI plots), `SVC` + Platt scaling, the preprocessing pipeline. |
| `xgboost` | 3.4.1 | Pairs with `shap` for the `TreeExplainer` output-space behaviour above. |
| `torch` | 2.13.0+cpu | `TorchMLPClassifier`; the CPU build, seeded via `torch.use_deterministic_algorithms`. |
| `pandas` | 3.0.4 | `DataFrame.applymap` was removed here - the reason `agreement.py`'s port of the reference `agreement_fraction` uses `.astype(str)`. |
| `numpy` / `scipy` | 2.4.6 / 1.18.0 | `rankdata(method="dense")` and the SEM used by the agreement metrics. |

## Datasets

Four intrusion detection datasets are supported: UNSW-NB15, CIC-IDS2017, DAPT2020, and NSL-KDD. Each
has a dataset adapter in `src/xai_thesis/datasets/` implementing dataset-specific cleanup (e.g.
deduplication, fixing known data-quality issues, dropping identifier/leakage columns) behind a shared
interface.

> **DAPT2020 and NSL-KDD are out of scope for the thesis** (DAPT2020 since 2026-07-31, NSL-KDD since
> 2026-08-13) **and were archived as of 2026-08-17.** Their adapter code (`src/xai_thesis/datasets/`)
> and tests remain fully functional and unmodified - nothing was deleted - but their raw/processed data
> moved to `data/_archive/<dataset>/`, their trained models/metrics/plots moved to
> `outputs/_archive/<dataset>/`, and their dataset/experiment configs moved to `configs/_archive/`. The
> thesis now covers only **UNSW-NB15 and CIC-IDS2017**, and the default (no-args) pipeline commands only
> ever touch those two. To rerun an archived experiment: `python scripts/train.py --experiment
> configs/_archive/experiments/nsl_kdd_random_forest_baseline.yaml --dataset-config-dir
> configs/_archive/datasets` (the archived dataset configs' `raw_path`/`processed_path` already point at
> the new data locations).

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
this project's own data first: it's `"unknown"` (the raw `-`/blank placeholder) for 57.33% of rows
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
into one `Web Attack` category. DAPT2020 is multiclass (its five kill-chain stages). NSL-KDD is kept
multiclass, but not at its native ~40-way specific attack-type granularity: it's grouped into the
conventional **DoS/Probe/R2L/U2R categories** (+ `Benign`) used throughout NSL-KDD literature -
structurally similar to DAPT2020's multi-stage setup.

UNSW-NB15's `Worms` category is severely data-scarce - **171 rows in the processed data (120 train / 51
test)** under the stratified split; the "174" figure sometimes quoted elsewhere is the pre-dedup raw
count. By far the rarest class in this dataset, in the same spirit as DAPT2020's `Data Exfiltration` or
NSL-KDD's `U2R`. For scale, `Benign` is **95.6%** of both the UNSW-NB15 train and test splits
(CIC-IDS2017: 84.9% `Benign`, `Infiltration` just 24 train / 11 test rows). UNSW-NB15's dominant error
pattern also doesn't funnel into `Benign` at all, unlike the other datasets: `Analysis`↔`Backdoor` is a
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
4 identical in every row (e.g. `Total Fwd Packets`/`Subflow Fwd Packets` differ in 0 of 2.83M rows), 4
more at r >= 0.99999 (differing in 1-54 rows), and `RST`/`ECE Flag Count` at r ~= 0.998 - see
`configs/datasets/cic_ids2017.yaml`'s `drop_columns` comment for the exact pairs. Also found in the same
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

One cleaning step was itself informed by the explainability analysis, not decided upfront: **DAPT2020**
drops `Src Port`/`Dst Port` alongside its IP/timestamp columns, since SHAP and permutation importance
surfaced suspicious port dominance (e.g. `Lateral Movement` driven by two fixed ports) that turned out to
be a testbed-specific identity leak rather than a generalizable signal. Retraining after the fix produced
measurably more coherent explanations (SHAP-vs-permutation-importance agreement improved), not just
better metrics.

**The same port-leak pattern resurfaced in CIC-IDS2017 itself, 2026-08-17/18** - user-flagged this time
("Destination Port... regularly appears in the explainability plots, but I believe it is an identifier
that needs to be removed"), then verified directly before acting on it: `Destination Port` was carrying
34% of `random_forest`'s total feature importance alone (>2.5x the next-highest feature, the #1 SHAP
feature for every class), and per-class port concentration was near-total - `DoS/DDoS` 99.999% port 80,
`Web Attack` 100% port 80, `Infiltration` 100% port 444, `Brute Force` 81%/19% port 21/22, `Bot` 55% port
8080 - because the CICIDS2017 testbed launched each attack scenario against one fixed target service, the
same kind of testbed-identity leak as DAPT2020's `Src Port`/`Dst Port` above, not generalizable traffic
behavior. A broader per-class value-concentration scan across the other 57 columns found nothing else
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
still does. CIC-IDS2017 and UNSW-NB15 ship without one, so they get a **stratified random split**
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
`nsl_kdd` is unaffected (its own official split has no ratio parameter).

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
> metrics/plot file (for UNSW-NB15, CIC-IDS2017, and their bundled copies within the archived
> NSL-KDD/DAPT2020 datasets), and now-dead explanation code (`logistic_regression`'s dedicated
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

All four datasets are severely class-imbalanced (NSL-KDD's `U2R` category has only 52 training rows and
67 test rows, against 67,343/9,711 `Benign`; UNSW-NB15 is 95.6% `Benign`, CIC-IDS2017 84.9%). **All four
models handle this by reweighting the training objective with the same per-class weight
`w_c = n / (k * count_c)` (scikit-learn's `class_weight="balanced"` formula), computed on the training
split only. None of them resample rows.**

The weight *values* are identical across models -
`sklearn.utils.class_weight.compute_sample_weight("balanced", y)` is literally the per-sample
broadcast of `class_weight="balanced"`, and both equal `n / (k * count_c)` (verified numerically; the
weights average to 1). What differs is only how each objective consumes them: weighted impurity/leaf
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

**Final results — stratified random 70/30 split, all four models re-tuned
(`RandomizedSearchCV`, `f1_macro`) and retrained, `svm` final fit on the 200k /
`min_per_class=500` subsample. Regenerated 2026-08-31/09-01.**

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
  0.43/0.46 once `learning_rate_init` was searched over a lower range. `xgboost` was already at its
  ceiling: re-tuning on the stratified split produced marginally *worse* params (0.893 / 0.554), so
  it keeps the earlier `RandomizedSearchCV` result (0.901 / 0.597) - the best of the two tuned
  configs, same "tuned, keep the best" rule as every other model.
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

Plots are written to `outputs/plots/<dataset_name>/<model>_baseline/confusion_matrix.png`.

Aggregate one-vs-rest ROC curves - macro- and micro-averaged, plus the chance diagonal, with each AUC in
the legend - for a trained experiment, or all of them:

```powershell
python scripts/roc_curve.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml
python scripts/roc_curve.py  # all experiments
```

Written to `outputs/plots/<dataset_name>/<model>_baseline/roc_curves.png`. Deliberately aggregated (no
per-class curves - unreadable at 7-10 classes, and that detail is already in the confusion matrix and
per-class F1): the macro-average AUC equals the `roc_auc_macro` in `outputs/metrics/` and is the one to
read; the micro-average is `Benign`-dominated and hugs the top-left corner regardless of rare-class
separation - the two are shown together to make that divergence visible.

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

Missing-value summary, train/test split balance, and feature correlation, for a dataset or all
datasets:

```powershell
python scripts/dataset_diagnostics.py --dataset unsw_nb15
python scripts/dataset_diagnostics.py  # all datasets
```

Plots are written alongside `label_distribution.png` in the same per-dataset folder:
`outputs/plots/<dataset_name>/missing_values.png`, `train_test_split.png`, and
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
  it. The multiplier is the knob to lower if the `svm` permutation path becomes intractable - cost
  scales ~linearly in it (CIC-IDS2017 `svm`: ~48 s/instance at 4 permutations, ~430 s at 50).
  Produces a global bar plot (mean |SHAP value| per feature, stacked by class), a global beeswarm
  for the class each model handles worst, and local waterfall plots for two curated test instances.
- **LIME** (`LimeTabularExplainer`): local explanations for the same two curated instances, so SHAP and
  LIME can be compared directly on identical cases. Rendered via `explain_instance_with_lime` in
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
  ones. Rendered as a hand-built horizontal bar chart, unlike the SHAP/LIME plots above - checked directly
  against `sklearn.inspection`'s actual contents (2026-08-28) and confirmed there is no native display
  class for this technique the way `PartialDependenceDisplay` exists for PDP, so a custom chart is the
  established way to visualize this, not a deviation from one.

The two curated instances per experiment (one correctly classified, one misclassified) are **not**
random: `generate_explanations()` anchors on the class *this trained model* handles worst - the
lowest per-class recall on its own test predictions - and the class it is most often mistaken for
(`_hardest_class_and_confusion` / `_most_common_confusion` in `pipeline.py`). Deriving both ends from
the model's own predictions replaced a hand-maintained `_CLASS_OF_INTEREST` dict (one curated pair per
dataset) that had to be re-verified by eye against every retrain - and after the stratified-split
re-tune its CIC-IDS2017 entry (`DoS/DDoS`→`Benign`) had dissolved to ~0% confusion while still sitting
in the config. On the current models the hardest class is a rare attack category on both datasets:
`DoS` or `Analysis`/`Backdoor` for UNSW-NB15 (where `Analysis`↔`Backdoor` remains a strong two-way
confusion, ~44-69% each way across the tree/NN models), `Infiltration` or `Bot` for CIC-IDS2017.

Plots are written to `outputs/plots/<dataset_name>/<model>_baseline/`.

### Limitation: SHAP base-value footing across models

Class balancing is no longer a source of cross-model inconsistency - all four models reweight the
objective with the same `n / (k * count_c)` weights and train on the same rows (`svm` on its 200k
final-fit subsample, the others on the full training set). Two explainer-mechanics points remain, worth
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
- **The global sample is class-balanced, not prevalence-weighted.** The global SHAP bar/beeswarm and
  permutation importance explain a **500-row class-balanced** test sample (`_class_balanced_sample`,
  `random_state=42`) - 50 rows/class on UNSW-NB15 (all 10 classes), 71/class on CIC-IDS2017 except
  `Infiltration`, which has only 11 test rows and contributes all of them (437 rows total). This
  replaced a 200-row *stratified* draw that was 191/200 `Benign` on UNSW-NB15 and left 7 of 10 classes
  with zero rows, making the global rankings a two-class problem. The trade-off: the rankings no longer
  reflect deployment prevalence (documented separately in `label_distribution.png`), and the
  permutation-importance macro-F1 drop is measured under the rebalanced distribution - intentional, and
  noted in `generate_explanations()`. What switching the basis did to the global rankings is measured
  under Cross-method agreement below.

### Interview stimuli

A second, separate curated-instance mechanism (`generate_interview_stimuli` in `pipeline.py`), for
picking concrete SHAP/LIME examples to show expert-interview subjects. Unlike `generate_explanations()`
above (anchors on the class each model handles *worst*), this one picks classes a security
practitioner would recognize by name, so interview time isn't spent explaining an unfamiliar
label: `Reconnaissance`/`Exploits` for UNSW-NB15, `DoS/DDoS`/`Brute Force`/`Bot` for CIC-IDS2017
(`_INTERVIEW_STIMULUS_CLASSES` in `pipeline.py` — see its comment for the excluded classes, and for
why `PortScan` was dropped on 2026-09-01: removing the leaky `Destination Port` column collapsed it
under dedup from ~159k rows to ~2k, so it no longer samples real scanning traffic). For each
configured class the model's most common real misclassification target is derived dynamically from
the test set (`_most_common_confusion`) rather than hand-picked, then one correct and one
misclassified SHAP waterfall + LIME plot are generated via the same instance-explanation logic
`generate_explanations` uses (`_generate_instance_explanations`).

```powershell
python scripts/interview_stimuli.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml
python scripts/interview_stimuli.py  # all experiments
```

Plots are written to `outputs/plots/<dataset_name>/<model>_baseline/interview_<class>/` (e.g.
`outputs/plots/unsw_nb15/random_forest_baseline/interview_Reconnaissance/`). A class maps to no
plots at all if the model never misclassifies it in the test set (no stimulus pair possible); every
class in the current list was re-checked to have both a correct and a misclassified test instance
for all four models.

### Cross-method agreement

Two panels, both built from the same two top-k metrics of **Krishna et al., "The Disagreement
Problem in Explainable Machine Learning" (TMLR 2024)**, so local and global read on one scale:

| panel | pair | unit compared |
| --- | --- | --- |
| local | SHAP vs LIME | one attribution vector per *instance*, averaged over a shared sample |
| global | mean \|SHAP\| pooled across classes vs permutation importance | one *dataset-level* ranking per model |

The two metrics:

- **Feature agreement** - fraction of the top-`k` features (ranked by `|attribution|`) the two
  explanations have in common, `|topk(A) n topk(B)| / k`. The paper's interviewed practitioners
  (84%) named the top-feature set as the single most important thing they read off an explanation.
- **Rank agreement** - fraction of the top-`k` features that are both shared *and* given the same
  rank by both explanations (dense ranking over all features, so ties are handled). Stricter than
  feature agreement and `<=` it by construction.

`src/xai_thesis/agreement.py`'s `agreement_fraction` is a direct port of the authors' reference
implementation (`AI4LIFE-GROUP/disagreement-problem`, `experiments/tabular/metrics.py`), verified
numerically identical to it across 144 randomised comparisons (both metrics x six `k` values x
twelve matrices, including tie-heavy and near-identical cases) before the reference copy was
removed. The only edit is the original's `DataFrame.applymap(str)` written as `.astype(str)`
(`applymap` was removed in pandas 3.0). The paper's other two metrics - sign agreement and signed
rank agreement - are deliberately not implemented; this project reports only the two above. No
rank-correlation measure is reported either: the paper's own rank correlation is defined over a
user-supplied set of features of interest, this project has no such set, and reporting one number
per panel keeps local and global directly comparable.

```powershell
python scripts/agreement.py --dataset unsw_nb15            # both panels
python scripts/agreement.py --scope global                # global only
python scripts/agreement.py --scope global --basis stratified --size 200
python scripts/agreement.py                               # every dataset
python scripts/agreement.py --overwrite                   # recompute the caches
```

#### Local: SHAP vs LIME

**Only SHAP and LIME enter these metrics.** They are defined over two explanations *of the same
prediction*, so they need a per-instance attribution vector from each method; permutation importance
is global-only and has no per-prediction explanation to compare, so it is not part of this analysis.
For each instance both methods explain the same class - the `predict_proba` argmax, which is what
LIME's `top_labels=1` selects.

Krishna et al. draw one method-by-method heatmap per model. With the method pair fixed at SHAP/LIME
there would be one informative number per panel, so the heatmaps here vary **model (rows) x top-`k`
(columns)** instead, one pair of panels per dataset:

- `outputs/plots/<dataset>/local_agreement_feature.png`
- `outputs/plots/<dataset>/local_agreement_rank.png`
- `outputs/metrics/agreement/<dataset>_local_agreement.csv` - the same numbers plus standard errors.

`k` sweeps `1, 5, 10, 20` by default (`--k`). The paper varies `k` the same way and finds agreement
falls as `k` grows, and its practitioners reported looking at "the top 5 to 10 features" - but its
`k` is a far larger share of the feature space than copying its numbers would give here: its
headline `k = 5` is 5 of COMPAS's **7** features (71%), and it sweeps to 100%, framed as
25/50/75/100%. Five of this project's 40 (UNSW-NB15) / 59 (CIC-IDS2017) columns is 8-12%, so the
sweep runs to 20 - 50% of UNSW-NB15, 34% of CIC-IDS2017 - rather than stopping at 10 and squashing
the whole figure into the top quarter of the ranking. It stops short of the feature count because
feature agreement is 1.0 there by definition.

`k = 1` is a local-panel column only. There it is the mean of ~60 per-instance outcomes with a real
standard error - "how often do the two methods even name the same single most important feature?",
which is what an analyst reading the top driver of an alert cares about. On the global panel it
would be a single ranking pair and could only ever read 0.0 or 1.0, so **the global sweep starts at
5** (`5, 10, 20`).

Metrics are averaged over a **class-balanced test sample** of ~60 rows (exactly 60 on UNSW-NB15:
6 per class x 10; 56 on CIC-IDS2017: 8 per class x 7, the remainder dropped by integer division -
every class, `Infiltration` included, has more than 8 test rows, so none is row-capped at this size)
rather than the ~85-96%-`Benign` stratified sample the global plots use - the question is SHAP/LIME
consistency across attack types. The draw depends only
on the test split and the seed, never on the model, so **all four models are compared on the
identical instances**; the result is therefore a statement about the two methods, not about which
rows each model happened to get.

Computing the attribution matrices is the slow part (`svm`'s permutation SHAP is ~90 s/instance on
UNSW-NB15 and ~410 s on CIC-IDS2017, plus one LIME run per instance), so they are cached as
`outputs/metrics/agreement/attributions/<experiment>.npz` and the heatmaps redraw from the cache in
seconds.

**Feature-space reconciliation**: SHAP is computed in the transformed space and collapsed back onto
the raw columns (one-hot `proto`/`service`/`state` summed into one column each - see
`compute_shap_values`), which is exactly the space LIME already works in. The two name sets are
identical for both datasets (40 columns for UNSW-NB15, 59 for CIC-IDS2017; LIME drops no
zero-variance column on either), so `_align_attribution_columns` is a guard rather than a real
reconciliation - but a future dataset with a constant column would need it.

> **`svm` caveat.** `SVC.predict()` (one-vs-one voting) and `predict_proba`'s argmax disagree on 34
> of the 60 sampled UNSW-NB15 instances - a known `SVC` + Platt-scaling effect under class imbalance
> (0 disagreements for every other model). The agreement numbers stay valid, since SHAP and LIME both
> explain the `predict_proba`-argmax class and so always describe the *same* class as each other, but
> for those instances that class is not the one the pipeline's `.predict()` returns.


**Local agreement at k = 5** (mean +/- standard error over the shared instance set):

| model | UNSW-NB15 feature@5 | UNSW-NB15 rank@5 | CIC-IDS2017 feature@5 | CIC-IDS2017 rank@5 |
| --- | --- | --- | --- | --- |
| `random_forest` | 0.237 +/- 0.029 | 0.047 +/- 0.011 | 0.261 +/- 0.029 | 0.054 +/- 0.013 |
| `xgboost` | 0.277 +/- 0.025 | 0.113 +/- 0.015 | 0.182 +/- 0.019 | 0.036 +/- 0.011 |
| `neural_network` | 0.520 +/- 0.023 | 0.150 +/- 0.017 | 0.379 +/- 0.033 | 0.064 +/- 0.014 |
| `svm` | 0.570 +/- 0.029 | 0.287 +/- 0.029 | *omitted* | *omitted* |

`svm` on CIC-IDS2017 is the one cell computed for no panel, local or global. Its permutation
SHAP was measured at **408 s/instance** (10-instance run at `_SHAP_PERMUTATION_MULTIPLIER = 50`,
2026-09-09), which is 6.8 h for the 60-instance local set and ~57 h for the 500-row global one,
against ~90 s/instance for the same model on UNSW-NB15. The pair is listed in
`_AGREEMENT_EXCLUDED` in `pipeline.py` with that measurement as its reason, dropped from the
heatmap rows, and named in each panel's footnote, so the omission is visible in the figure
rather than only here. Lowering the multiplier for `svm` alone was rejected: an inconsistent
permutation count across models would make the cross-model comparison meaningless.

#### Global: SHAP vs permutation importance

`generate_global_agreement` applies the *same two metrics* to the two global rankings a model
produces: mean `|SHAP|` pooled across classes (what `shap_bar.png` stacks) and
`permutation_importance(...).importances_mean` (what `permutation_importance.png` plots). Each cell
is a single number rather than a mean over instances, because a global ranking is one vector per
model - so these panels carry no standard error.

The two rankings are **not recomputed** for the figure. `generate_explanations()` caches exactly the
pair it plotted into `outputs/metrics/agreement/global/<experiment>__<basis><size>.npz`, and the
agreement panel reads that back, so the figure describes the ranking the thesis actually shows
rather than an independent re-run of it - and `svm`, at ~90 s/instance over 500 rows, is computed
once instead of twice.

Permutation importance goes slightly negative when permuting a feature happens to *improve* the
score. Since the metrics rank by `|value|`, those are clipped to zero first
(`_as_importance_vector`): a negative permutation importance means "no measurable importance", not
"strongly important the other way", and without the clip a feature scoring `-0.004` would outrank
one scoring `+0.001`.

Outputs:

- `outputs/plots/<dataset>/global_agreement_feature.png`
- `outputs/plots/<dataset>/global_agreement_rank.png`
- `outputs/metrics/agreement/<dataset>_global_agreement.csv` - the same numbers plus each model's
  top-5 features under both methods.

**One standardised figure set.** Across both panels and both datasets that is **eight heatmaps** -
`local_agreement_{feature,rank}.png` and `global_agreement_{feature,rank}.png` per dataset - all on the
class-balanced 500-row basis, `random_state=42`, the same two metrics and the same fixed `[0, 1]`
sequential-`Reds` scale (light = low agreement, dark = high), with models as rows. The scale is
sequential rather than diverging because agreement runs 0 to 1 with no meaningful midpoint - a
diverging palette's white band at 0.5 would imply a neutral point that does not exist - and it
matches the paper's own convention (Figure 1: "lower values (lighter colors) indicating stronger
disagreement"). An alternative `--basis`/`--size` writes **only** a CSV, suffixed
`_<basis><size>`; it does not draw a heatmap unless `write_plots=True` is passed explicitly. The
alternative bases exist to justify the default (below), and are reported as numbers - a near-identical
heatmap per basis would bury the figure the thesis actually uses.

**Sample basis.** `--basis`/`--size` select the rows the rankings are built on, which is how the
2026-09-09 switch away from a prevalence-proportional draw is reproduced:

| basis | draw | what used it |
| --- | --- | --- |
| A | stratified, 200 rows | the global plots before 2026-09-09 (191/200 `Benign` on UNSW-NB15) |
| B | class-balanced, 60 rows | the local panel's instance set, reused as a global basis |
| C | class-balanced, 500 rows | **current default** - the sample `generate_explanations()` plots |

**Global agreement at k = 5, by basis** (`random_forest` also run at 1000 rows as a
convergence check; `svm` under the default basis only - the alternative bases were a
methodological comparison and are not worth 5 h and 1.5 h of its permutation SHAP):

*UNSW-NB15*

| model | basis | rows | classes | feature@5 | rank@5 | SHAP top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| `random_forest` | A stratified, 200 | 200 | 6 | 0.80 | 0.40 | sttl, ct_state_ttl, sbytes, service, proto |
|  | B balanced, 60 | 60 | 10 | 0.80 | 0.20 | proto, service, sbytes, sttl, smean |
|  | C **balanced, 500** | 500 | 10 | 0.80 | 0.40 | proto, service, sbytes, sttl, dbytes |
|  | C' balanced, 1000 | 951 | 10 | 0.80 | 0.40 | proto, service, sbytes, sttl, dbytes |
| `xgboost` | A stratified, 200 | 200 | 6 | 0.60 | 0.20 | sttl, proto, dbytes, sbytes, dmean |
|  | B balanced, 60 | 60 | 10 | 1.00 | 0.20 | proto, sbytes, sttl, service, smean |
|  | C **balanced, 500** | 500 | 10 | 1.00 | 0.20 | proto, sttl, sbytes, service, smean |
| `neural_network` | A stratified, 200 | 200 | 6 | 0.60 | 0.40 | sttl, dttl, state, service, proto |
|  | B balanced, 60 | 60 | 10 | 1.00 | 0.40 | dttl, service, proto, sttl, state |
|  | C **balanced, 500** | 500 | 10 | 1.00 | 0.40 | dttl, service, sttl, proto, state |
| `svm` | C **balanced, 500** | 500 | 10 | 0.80 | 0.00 | dttl, proto, state, service, dmean |

*CIC-IDS2017*

| model | basis | rows | classes | feature@5 | rank@5 | SHAP top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| `random_forest` | A stratified, 200 | 200 | 3 | 0.60 | 0.20 | Bwd Packet Length Std, Init_Win_bytes_backward, Init_Win_bytes_forward, Fwd IAT Std, Bwd Packet Length Mean |
|  | B balanced, 60 | 56 | 7 | 0.80 | 0.40 | Fwd Packet Length Max, Fwd IAT Std, Total Backward Packets, Bwd Packet Length Std, Init_Win_bytes_backward |
|  | C **balanced, 500** | 437 | 7 | 0.80 | 0.20 | Fwd IAT Std, Total Backward Packets, Bwd Header Length, Bwd Packet Length Std, Fwd Packet Length Max |
|  | C' balanced, 1000 | 863 | 7 | 0.80 | 0.20 | Fwd IAT Std, Total Backward Packets, Bwd Header Length, Bwd Packet Length Std, Init_Win_bytes_backward |
| `xgboost` | A stratified, 200 | 200 | 3 | 0.40 | 0.00 | Init_Win_bytes_forward, Init_Win_bytes_backward, Flow IAT Min, min_seg_size_forward, Bwd Packets/s |
|  | B balanced, 60 | 56 | 7 | 0.40 | 0.00 | Init_Win_bytes_forward, Init_Win_bytes_backward, Bwd Packets/s, Flow IAT Min, min_seg_size_forward |
|  | C **balanced, 500** | 437 | 7 | 0.60 | 0.40 | Init_Win_bytes_backward, Init_Win_bytes_forward, Bwd Packets/s, Flow IAT Min, Fwd Packets/s |
| `neural_network` | A stratified, 200 | 200 | 3 | 0.40 | 0.40 | PSH Flag Count, Init_Win_bytes_forward, ACK Flag Count, min_seg_size_forward, Bwd IAT Total |
|  | B balanced, 60 | 56 | 7 | 0.80 | 0.60 | PSH Flag Count, Init_Win_bytes_forward, ACK Flag Count, Init_Win_bytes_backward, min_seg_size_forward |
|  | C **balanced, 500** | 437 | 7 | 0.80 | 0.20 | PSH Flag Count, Init_Win_bytes_forward, Init_Win_bytes_backward, ACK Flag Count, min_seg_size_forward |

Reading the table:

- **A -> C is a change of object, not just of score.** The stratified draw only ever sees 6 of
  UNSW-NB15's 10 classes and 3 of CIC-IDS2017's 7, so its "global" ranking describes a handful of
  classes. Feature agreement rises in 5 of the 6 model/dataset pairs on the switch, and features
  that lead under A - UNSW-NB15's `ct_state_ttl`, CIC-IDS2017's `Bwd Packet Length Std` - leave the
  top 5 entirely under C.
- **B -> C moves rankings, which is why C' exists.** Feature agreement is mostly stable, but three
  of six models swap a top-5 member between the 60-row and 500-row balanced draws
  (`smean`->`dbytes`, `Init_Win_bytes_backward`->`Bwd Header Length`,
  `min_seg_size_forward`->`Fwd Packets/s`), and CIC-IDS2017's `random_forest` reorders heavily.
- **500 rows is converged.** Doubling to ~1000 (`random_forest`, both datasets) reproduces every
  feature- and rank-agreement value at every `k` on UNSW-NB15, and every value on CIC-IDS2017 except
  rank agreement at `k = 10`; the only top-5 movement is CIC-IDS2017's fifth-ranked feature. The
  B -> C movement was therefore small-sample noise at n = 56-60, not a sign that 500 is still short.
- **Global rank agreement is a single ranking pair, so it does not average.** At `k = 5` it can only
  take values in steps of 0.2, and one feature shifting one position moves it a whole step. Feature
  agreement is the more stable read at this scale; the local panel, averaged over ~60 instances, is
  where rank agreement carries a meaningful standard error.
