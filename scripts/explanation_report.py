"""Export the tables behind the thesis's global/local explanation and disagreement results.

Reads only existing caches (see src/xai_thesis/explanation_report.py) and writes CSVs to
outputs/metrics/explanation_report/:

  <dataset>_global_top10.csv          top-10 features with values, per model and method
                                      (mean |SHAP| and permutation importance)
  <dataset>_global_overlap.csv        feature/rank agreement at k = 5, 10, 20: SHAP vs PFI per model
                                      ("methods"), and every model pair per method ("models")
  <dataset>_global_disagreement.csv   features in one method's top 10 but not the other's, with
                                      their rank under both methods
  <dataset>_local_instances.csv       the agreement instance set: true/predicted/explained class and
                                      per-instance feature/rank/sign agreement
  <dataset>_local_top10.csv           signed SHAP and LIME top-10 per instance
  <dataset>_local_disagreement.csv    per instance: top-10 membership disagreements and sign flips
  <dataset>_local_summary.csv         per model, over all / correct / misclassified instances

Run scripts/agreement.py first if a cache is missing; this script never computes SHAP or LIME.
"""

from __future__ import annotations

import argparse

from xai_thesis.explanation_report import build_global_report, build_local_report
from xai_thesis.pipeline import _AGREEMENT_MODEL_ORDER

_DATASETS = ("unsw_nb15", "cic_ids2017")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=_DATASETS, help="Default: both datasets.")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--output-dir", default="outputs/metrics/explanation_report")
    args = parser.parse_args()

    for dataset_name in [args.dataset] if args.dataset else _DATASETS:
        paths = {
            **build_global_report(
                dataset_name, _AGREEMENT_MODEL_ORDER, metrics_dir=args.metrics_dir, output_dir=args.output_dir
            ),
            **build_local_report(
                dataset_name, _AGREEMENT_MODEL_ORDER, metrics_dir=args.metrics_dir, output_dir=args.output_dir
            ),
        }
        for name, path in paths.items():
            print(f"{dataset_name} {name}: {path}")


if __name__ == "__main__":
    main()
