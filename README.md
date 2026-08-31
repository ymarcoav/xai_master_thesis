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
own, not from that paper: verified to correlate with `is_ftp_login` at r≈0.85 (near-redundant with a
feature already kept) and, when nonzero (only ~1.7% of rows), to show no per-class differentiation -
its mean value is ~1.0-1.2 across every `attack_cat` with any nonzero rows at all.

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

UNSW-NB15's `Worms` category is severely data-scarce - **171 rows in the processed data (118 train / 53
test)**; the "174" figure sometimes quoted elsewhere is the pre-dedup raw count. By far the rarest class
in this dataset, in the same spirit as DAPT2020's `Data Exfiltration` or NSL-KDD's `U2R`. For scale,
`Benign` is **95.6%** of both the UNSW-NB15 train and test splits (CIC-IDS2017: 84.9% `Benign`,
`Infiltration` just 24 train / 11 test rows). Its dominant confusion pattern also doesn't funnel into `Benign` at all,
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
up best, still noticeably degraded). The dominant cross-model confusion pair didn't change
(`DoS/DDoS`→`Benign`, see `_CLASS_OF_INTEREST` below) but got dramatically more pronounced - 68.9-89.3%
now, up from 6.3-13.2% before - itself strong independent evidence that the old numbers were substantially
propped up by the port leak rather than reflecting genuine separability.

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
> Every processed dataset, trained model, and result number below predates this change and must be
> regenerated - rerun `scripts/preprocess.py`, then the training and explainability steps.

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
model's contract requires (`evaluate_classifier`, `compute_shap_values`, `_CLASS_OF_INTEREST`'s
string lookups, etc. all assume this). It's a real subclass, not a wrapper, specifically so
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

> **STALE as of the 2026-08-31 split change (see Preprocessing).** Every number in this section was
> produced under the old per-(source_file, label) chronological/positional hold-out. It is kept here
> only for reference until the stratified-random-split retrain lands; do not cite it as current.

**Final results, both datasets, all four models, on the corrected raw data / 70-30 split / (for `svm`)
the 200k/`min_per_class=500` subsample:**

| Model | UNSW-NB15 macro-F1 | UNSW-NB15 accuracy | CIC-IDS2017 macro-F1 | CIC-IDS2017 accuracy |
| --- | --- | --- | --- | --- |
| `xgboost` | **0.576** | 0.972 | **0.787** | 0.988 |
| `random_forest` | 0.572 | 0.972 | 0.630 | 0.958 |
| `neural_network` | 0.449 | 0.964 | 0.374 | 0.822 |
| `svm` | 0.411 | 0.965 | 0.371 | 0.822 |

`svm` did **not** end up outperforming `neural_network` on UNSW-NB15 (0.411 vs 0.449, essentially
unchanged by the 150k/200 -> 200k/500 raise - 0.4126 -> 0.4112, a wash) - a linear-kernel SVM landing
last among four models on a severely imbalanced 10-class problem with real class-overlap (see
`Analysis`<->`Backdoor` above) isn't overturned by subsample tuning alone. On CIC-IDS2017, `svm` and
`neural_network` landed close enough to call a tie (0.371 vs 0.374) - once the `max_iter` bug was fixed,
the larger subsample's macro-F1 (0.371) is a real improvement over the original 150k/200 run's 0.293,
even though it didn't reach `neural_network`. `svm` remains the weakest model on both datasets; `xgboost`
is the strongest on both.

**`xgboost` added 2026-08-24, in scope for the thesis (see the blockquote above).** Tuned
(`RandomizedSearchCV`, `subsample_size=40_000` for tractability on both large datasets, same
pattern already used for `svm` above) and trained on the complete, un-subsampled training set
for both - no `_FINAL_FIT_SUBSAMPLE` exception needed, unlike `svm`: `tree_method="hist"` finishes
CIC-IDS2017's final fit (500 trees) in well under 4 minutes regardless of dataset size. See the results
table above for current numbers.

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

- **SHAP**: `TreeExplainer` for `random_forest` and `xgboost`, and a permutation-based `Explainer`
  for `neural_network` and `svm` (no fast structural explainer exists for a plain MLP or a kernel SVM -
  this path is markedly slower for both, and slower still for `svm` specifically, since its own
  predictions are more expensive to compute).
  Produces a global bar plot (mean |SHAP value| per feature, stacked by class), a global beeswarm
  for the dataset's class of interest, and local waterfall plots for two curated test instances.
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
- **Permutation feature importance** (`sklearn.inspection.permutation_importance`): run on the full
  pipeline, so importances are reported in terms of the original feature names, not the one-hot-expanded
  ones. Rendered as a hand-built horizontal bar chart, unlike the SHAP/LIME plots above - checked directly
  against `sklearn.inspection`'s actual contents (2026-08-28) and confirmed there is no native display
  class for this technique the way `PartialDependenceDisplay` exists for PDP, so a custom chart is the
  established way to visualize this, not a deviation from one.

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
- **The rarest classes are near-absent from the global plots.** The global SHAP bar/beeswarm and
  permutation importance explain a 200-row **stratified** (prevalence-proportional) test sample - a
  deliberate "what drives predictions on realistic traffic" choice - so `Worms` / `Infiltration` have
  ~0 rows in it and their stacked segment in `shap_bar.png` is low-support. Rare classes are examined
  only through the per-class local explanations (curated instances, interview stimuli).

### Interview stimuli

A second, separate curated-instance mechanism (`generate_interview_stimuli` in `pipeline.py`), for
picking concrete SHAP/LIME examples to show expert-interview subjects. Unlike `_CLASS_OF_INTEREST`
above (picks the class pair with the most significant cross-model *confusion*), this one picks classes
a security practitioner would recognize by name, so interview time isn't spent explaining an unfamiliar
label: `Reconnaissance`/`Exploits` for UNSW-NB15, `DoS/DDoS`/`PortScan`/`Brute Force` for CIC-IDS2017
(`_INTERVIEW_STIMULUS_CLASSES` in `pipeline.py` — see its comment for why each excluded class,
including the well-known `DoS`/`Bot`, was ruled out). For each configured class, the model's most
common real misclassification target is derived dynamically from the test set (`_most_common_confusion`)
rather than hand-picked, then one correct and one misclassified SHAP waterfall + LIME plot are generated
via the same instance-explanation logic `generate_explanations` uses (`_generate_instance_explanations`).

```powershell
python scripts/interview_stimuli.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml
python scripts/interview_stimuli.py  # all experiments
```

Plots are written to `outputs/plots/<dataset_name>/<model>_baseline/interview_<class>/` (e.g.
`outputs/plots/unsw_nb15/random_forest_baseline/interview_Reconnaissance/`). A class maps to no
plots at all if the model never misclassifies it in the test set (no stimulus pair possible - e.g.
CIC-IDS2017's `Bot` was excluded from the configured list for exactly this reason on two models).
