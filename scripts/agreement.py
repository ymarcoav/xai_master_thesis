"""SHAP vs LIME local agreement heatmaps for a dataset's models.

Feature agreement and rank agreement (Krishna et al. 2022 - see src/xai_thesis/agreement.py),
per instance, averaged over a shared class-balanced test sample (total=60, random_state=42).
This is the project's systematic agreement analysis; globally, only the SHAP vs PFI top-10
overlap is reported (scripts/explanation_report.py).

Separate from scripts/explain.py because the per-instance attribution matrices are slow for
svm (permutation SHAP is ~90 s/instance on UNSW-NB15). They are cached under
outputs/metrics/agreement/attributions/, so redrawing afterwards is instant; pass --overwrite to
recompute.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from xai_thesis.pipeline import generate_agreement_heatmaps
from xai_thesis.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        help="Dataset name (e.g. unsw_nb15). If omitted, every dataset with experiment configs is used.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/experiments",
        help="Directory containing experiment YAML configs.",
    )
    parser.add_argument(
        "--dataset-config-dir",
        default="configs/datasets",
        help="Directory containing dataset YAML configs.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="Top-k values for the heatmap columns (default: 1 5 10).",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=60,
        help="Class-balanced test instances the local metrics are averaged over.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Restrict the panel to these models (default: every model the dataset has, minus "
        "any on the permanent exclusion list).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute the cached attributions instead of reusing them.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(
            {
                load_yaml(config_path)["dataset"]
                for config_path in Path(args.config_dir).glob("*.yaml")
            }
        )

    for dataset_name in datasets:
        kwargs = {
            "config_dir": args.config_dir,
            "dataset_config_dir": args.dataset_config_dir,
            "overwrite": args.overwrite,
            "models": args.models,
            "n_instances": args.n_instances,
        }
        if args.k:
            kwargs["k_values"] = args.k
        paths = generate_agreement_heatmaps(dataset_name, **kwargs)
        for kind, path in paths.items():
            print(f"{dataset_name} local: wrote {kind}: {path}")


if __name__ == "__main__":
    main()
