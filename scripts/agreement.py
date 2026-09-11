"""Cross-method explanation agreement heatmaps for a dataset's models.

Two panels, both using the same two top-k metrics from Krishna et al. 2022 (feature agreement
and rank agreement - see src/xai_thesis/agreement.py), so local and global read on one scale:

  local   SHAP vs LIME, per instance, averaged over a shared class-balanced test sample.
  global  mean |SHAP| pooled across classes vs permutation importance, one ranking pair per
          model, on the same sample generate_explanations() builds its global plots from.

Separate from scripts/explain.py because the per-instance attribution matrices are slow for
svm (permutation SHAP is ~90 s/instance on UNSW-NB15). Both scopes cache what they compute
under outputs/metrics/agreement/, so redrawing afterwards is instant; pass --overwrite to
recompute. --basis/--size vary the global sample, which is how the switch away from a
prevalence-proportional draw is reproduced (e.g. --basis stratified --size 200).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_thesis.pipeline import generate_agreement_heatmaps, generate_global_agreement
from xai_thesis.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        help="Dataset name (e.g. unsw_nb15). If omitted, every dataset with experiment configs is used.",
    )
    parser.add_argument(
        "--scope",
        choices=("local", "global", "both"),
        default="both",
        help="Which panel to build (default: both).",
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
        help="Top-k values for the heatmap columns (default: 1 5 10 20 local, 5 10 20 global).",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=60,
        help="Class-balanced test instances the local metrics are averaged over.",
    )
    parser.add_argument(
        "--basis",
        choices=("balanced", "stratified"),
        default="balanced",
        help="How the global-ranking sample is drawn (default: balanced).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="Rows in the global-ranking sample (default: 500).",
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
        help="Recompute the cached attributions/rankings instead of reusing them.",
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

    shared = {
        "config_dir": args.config_dir,
        "dataset_config_dir": args.dataset_config_dir,
        "overwrite": args.overwrite,
        "models": args.models,
    }

    for dataset_name in datasets:
        if args.scope in ("local", "both"):
            local_kwargs = dict(shared, n_instances=args.n_instances)
            if args.k:
                local_kwargs["k_values"] = args.k
            paths = generate_agreement_heatmaps(dataset_name, **local_kwargs)
            for kind, path in paths.items():
                print(f"{dataset_name} local: wrote {kind}: {path}")
        if args.scope in ("global", "both"):
            global_kwargs = dict(shared, basis=args.basis, size=args.size)
            if args.k:
                global_kwargs["k_values"] = args.k
            paths = generate_global_agreement(dataset_name, **global_kwargs)
            for kind, path in paths.items():
                print(f"{dataset_name} global ({args.basis}-{args.size}): wrote {kind}: {path}")


if __name__ == "__main__":
    main()
