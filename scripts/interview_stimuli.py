"""Generate SHAP/LIME interview stimuli (well-known-class curated instances) for a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_thesis.pipeline import generate_interview_stimuli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        help="Path to an experiment YAML config. If omitted, all configs in --config-dir are used.",
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
    args = parser.parse_args()

    if args.experiment:
        config_paths = [Path(args.experiment)]
    else:
        config_paths = sorted(Path(args.config_dir).glob("*.yaml"))

    for config_path in config_paths:
        results = generate_interview_stimuli(config_path, dataset_config_dir=args.dataset_config_dir)
        for target_class, paths in results.items():
            if not paths:
                print(f"{config_path.stem}: skipped {target_class} (no misclassified example in test set)")
                continue
            for kind, path in paths.items():
                print(f"{config_path.stem} / {target_class}: wrote {kind}: {path}")


if __name__ == "__main__":
    main()
