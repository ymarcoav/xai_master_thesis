"""Feature-agreement and rank-agreement heatmaps (SHAP vs LIME) for a dataset's models.

Implements the two top-k disagreement metrics of Krishna et al. 2022; see
src/xai_thesis/agreement.py. Separate from scripts/explain.py because computing the
per-instance attribution matrices is slow for svm (permutation SHAP is ~90 s/instance on
UNSW-NB15, ~410 s on CIC-IDS2017). Those matrices are cached under
outputs/metrics/agreement/attributions/, so re-drawing the heatmaps afterwards is instant;
pass --overwrite to recompute them.
"""

from __future__ import annotations

import argparse
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
        help="Top-k values for the heatmap columns (default: 1 3 5 8 10).",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=60,
        help="Class-balanced test instances the metrics are averaged over.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute the cached SHAP/LIME attribution matrices instead of reusing them.",
    )
    args = parser.parse_args()

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
            "n_instances": args.n_instances,
            "overwrite": args.overwrite,
        }
        if args.k:
            kwargs["k_values"] = args.k
        paths = generate_agreement_heatmaps(dataset_name, **kwargs)
        for kind, path in paths.items():
            print(f"{dataset_name}: wrote {kind}: {path}")


if __name__ == "__main__":
    main()
