"""Export the tables behind the thesis's explanation results.

Reads only existing caches (see src/xai_thesis/explanation_report.py) and writes CSVs to
outputs/metrics/explanation_report/:

  local_agreement.csv                 SHAP vs LIME, per dataset/model/k (1, 5, 10): feature and
                                      rank agreement mean and standard error, and n
  global_top10.csv                    top-10 features per dataset, model and method (SHAP, PFI),
                                      with rank and importance
  global_overlap.csv                  per dataset and model: number of features the SHAP and PFI
                                      top 10s share (feature agreement at k = 10)
  sample_sizes.csv                    per-class row counts of the local and global samples, after
                                      checking every cache was computed on exactly those rows
  <dataset>_local_instances.csv       the agreement instance set: true/predicted/explained class and
                                      per-instance feature/rank (and supplementary sign) agreement
  <dataset>_local_top10.csv           signed SHAP and LIME top-10 per instance
  <dataset>_local_disagreement.csv    per instance: top-10 membership disagreements and sign flips
  <dataset>_local_summary.csv         per model, over all / correct / misclassified instances

Run scripts/agreement.py and scripts/explain.py first if a cache is missing; this script never
computes SHAP, LIME or permutation importance.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from xai_thesis.explanation_report import build_local_report, write_thesis_tables
from xai_thesis.pipeline import _AGREEMENT_MODEL_ORDER, sample_composition

_DATASETS = ("unsw_nb15", "cic_ids2017")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--output-dir", default="outputs/metrics/explanation_report")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    sample_sizes = pd.concat(
        [sample_composition(dataset_name, metrics_dir=args.metrics_dir) for dataset_name in _DATASETS],
        ignore_index=True,
    )
    paths = write_thesis_tables(
        _DATASETS, _AGREEMENT_MODEL_ORDER, sample_sizes, metrics_dir=args.metrics_dir, output_dir=args.output_dir
    )
    for dataset_name in _DATASETS:
        paths.update(
            {
                f"{dataset_name} {name}": path
                for name, path in build_local_report(
                    dataset_name, _AGREEMENT_MODEL_ORDER, metrics_dir=args.metrics_dir, output_dir=args.output_dir
                ).items()
            }
        )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
