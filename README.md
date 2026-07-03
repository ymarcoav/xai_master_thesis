# xai_master_thesis

Code for the master thesis titled "Evaluating Explainable AI Methods for Intrusion Detection Systems: From Model Explanations to Security Decisions".

## Structure

- `configs/`: dataset and experiment configuration.
- `data/`: raw and processed datasets. Data files are ignored by Git.
- `src/xai_thesis/`: reusable project code.
- `scripts/`: command-line entry points for preprocessing, training, and explanations.
- `notebooks/`: exploratory analysis and thesis figures in progress.
- `outputs/`: generated models, metrics, and plots.
- `tests/`: focused tests for shared code.

## Dataset Inspection

Inspect raw dataset files before implementing preprocessing:

```powershell
python scripts/inspect_datasets.py
```

For a quick sampled check:

```powershell
python scripts/inspect_datasets.py --max-rows 1000
```

Reports are written to `outputs/metrics/dataset_inspection/`.
