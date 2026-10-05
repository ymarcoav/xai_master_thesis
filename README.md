# Explainable AI for Network Intrusion Detection

Code for the master's thesis *Evaluating Explainable AI Methods for Intrusion Detection Systems:
From Model Explanations to Security Decisions*. It trains four class-weighted classifiers (random
forest, XGBoost, multi-layer perceptron, linear SVM) on UNSW-NB15 and CIC-IDS2017, explains them
with SHAP, LIME and permutation feature importance, and measures how far the explanations agree.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/xai_thesis/datasets/` | Dataset adapters: loading and cleaning of the raw files |
| `src/xai_thesis/preprocessing.py` | Train/test split and the preprocessing pipeline (imputation, scaling, encoding) |
| `src/xai_thesis/models.py` | The four models and their hyperparameter search spaces |
| `src/xai_thesis/evaluation.py` | Metrics, confusion matrices and data plots |
| `src/xai_thesis/explainability.py` | SHAP, LIME, permutation feature importance and their plots |
| `src/xai_thesis/agreement.py` | Feature and rank agreement (Krishna et al.) |
| `src/xai_thesis/pipeline.py` | End-to-end steps called by the scripts |
| `scripts/` | Command-line entry points, one per pipeline step |
| `configs/datasets/`, `configs/experiments/` | Dataset settings and the eight model configurations, including the selected hyperparameters |
| `tests/` | Unit and integration tests |
| `results/` | The metrics and tables reported in the thesis |

## Setup

Python 3.14 on Windows was used; `requirements.lock` pins the exact environment.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e . --no-deps
```

## Data

The datasets are not included. Place the original CSV files as follows:

| Dataset | Source | Files |
| --- | --- | --- |
| UNSW-NB15 | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> | `UNSW-NB15_1.csv` to `UNSW-NB15_4.csv` and `NUSW-NB15_features.csv` in `data/raw/UNSW_NB15/` |
| CIC-IDS2017 | <https://www.unb.ca/cic/datasets/ids-2017.html> (MachineLearningCVE) | the eight `*.pcap_ISCX.csv` files in `data/raw/CIC_IDS2017/` |

## Reproducing the results

Run the steps in order from the repository root. Each script processes all datasets or
configurations unless restricted with `--dataset` or `--experiment`.

```powershell
# 1. Cleaning, deduplication and the stratified 70/30 split (writes data/processed/)
python scripts/preprocess.py

# 2. Dataset figures and the redundancy screen
python scripts/label_distribution.py
python scripts/feature_screen.py

# 3. Training and test-set evaluation (add --tune to rerun the hyperparameter search first)
Get-ChildItem configs/experiments/*.yaml | ForEach-Object { python scripts/train.py --experiment $_.FullName }
python scripts/confusion_matrix.py
python scripts/roc_curve.py

# 4. Global explanations: SHAP and permutation feature importance
python scripts/explain.py

# 5. Local agreement between SHAP and LIME
python scripts/agreement.py

# 6. Report tables and the agreement figure
python scripts/stimulus_candidates.py
python scripts/explanation_report.py
python scripts/thesis_agreement_figure.py
```

On a 16-core desktop CPU, the random forest, XGBoost and the MLP train in minutes and the SVM in
up to a few hours. The SVM's SHAP values are estimated by sampling and dominate the run time: its
local agreement takes about 1.5 hours (UNSW-NB15) and 7 hours (CIC-IDS2017). Global SHAP for the
CIC-IDS2017 SVM is skipped by default (see its config); to compute it, set
`skip_global_shap: false` and run it in parallel chunks:

```powershell
python scripts/explain.py --experiment configs/experiments/cic_ids2017_svm_baseline.yaml --shap-chunk-size 5 --shap-n-jobs 8
```

Local SHAP and LIME plots of individual alerts are produced by `scripts/interview_stimuli.py`. The
stimuli used in the expert interviews were generated with an earlier configuration of the
CIC-IDS2017 XGBoost model, as described in the thesis.

## Where the thesis results come from

| Thesis content | Script | Output |
| --- | --- | --- |
| Class distributions (Datasets section, Appendix) | `label_distribution.py`, `preprocess.py` | `outputs/plots/<dataset>/data/`, `outputs/metrics/dataset_inspection/` |
| Constant and near-duplicate columns | `feature_screen.py` | `outputs/metrics/dataset_inspection/<dataset>/feature_screen.csv` |
| Performance tables | `train.py` | `outputs/metrics/<experiment>.json` |
| Confusion matrices | `confusion_matrix.py` | `outputs/plots/<dataset>/<model>_baseline/performance/` |
| Global explanations and top-10 overlap | `explain.py`, `explanation_report.py` | `outputs/plots/<dataset>/<model>_baseline/global/`, `global_top10.csv`, `global_overlap.csv` |
| Local agreement between SHAP and LIME | `agreement.py`, `explanation_report.py`, `thesis_agreement_figure.py` | `local_agreement.csv`, `outputs/plots/thesis_figures/` |

The report tables are written to `outputs/metrics/explanation_report/`. `results/` holds a copy
of the metrics and tables as reported in the thesis.

## Method summary

- **Cleaning:** label normalization (CIC-IDS2017: seven consolidated classes); infinite values
  treated as missing; removal of corrupted records, identifiers, timestamps, constant and
  near-duplicate columns, and exact duplicate records. The dataset-specific steps are documented
  in `src/xai_thesis/datasets/` and `configs/datasets/`.
- **Preprocessing:** stratified 70/30 split; median imputation, min-max scaling and one-hot
  encoding (categories below 1% grouped), all fitted on the training partition.
- **Models:** balanced class weights for all four models. Hyperparameters are selected by
  randomized search (up to 30 configurations) with stratified 3-fold cross-validation and
  macro-F1 on a 200,000-row class-proportional subsample of the training partition. The SVM's
  final fit uses a stratified subsample of about 200,000 rows with at least 500 per class where
  available.
- **Explanations:** TreeExplainer for the tree ensembles (exact) and the permutation explainer for
  the SVM and the MLP (sampled); LIME on the predicted probabilities; permutation importance as the
  macro-F1 drop over 10 repetitions. Global explanations use a class-balanced test sample (500 and
  437 instances), the agreement analysis another one (60 and 56 instances).
- **Reproducibility:** all random components use the seed 42.

## Tests

```powershell
python -m pytest
```
