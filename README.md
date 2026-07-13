# xai_master_thesis

Code for the master thesis titled "Evaluating Explainable AI Methods for Intrusion Detection Systems: From Model Explanations to Security Decisions".

## Structure

- `configs/datasets/`: one YAML per dataset (raw/processed paths, label column, columns to drop, categorical columns).
- `configs/experiments/`: one YAML per trained model (dataset, model type, hyperparameters, metrics to report).
- `data/`: raw and processed datasets. Data files are ignored by Git.
- `src/xai_thesis/`: reusable project code (dataset adapters, preprocessing, models, evaluation, pipeline orchestration).
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
(`random_forest`, `svm`, `neural_network`, or `gradient_boosting`). Train one:

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

## Analysis

Confusion matrix (row-normalized by default, since class counts are severely imbalanced) for a
trained experiment, or all experiments in `configs/experiments/`:

```powershell
python scripts/confusion_matrix.py --experiment configs/experiments/unsw_nb15_random_forest_baseline.yaml
python scripts/confusion_matrix.py  # all experiments
```

Label class distribution for a dataset, or all datasets:

```powershell
python scripts/label_distribution.py --dataset unsw_nb15
python scripts/label_distribution.py  # all datasets
```

Plots are written to `outputs/plots/`.

## Explainability

Not yet implemented (`scripts/explain.py`, `src/xai_thesis/explainability.py` are stubs) — pending a
decision on which explainability methods and plots to use.
