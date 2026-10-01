"""Generate global explainability artifacts (SHAP bar, PFI) for a trained model.

Local SHAP/LIME plots are interview stimuli only - see scripts/interview_stimuli.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_thesis.pipeline import generate_explanations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        help="Path to an experiment YAML config. If omitted, all configs in --config-dir are explained.",
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
        paths = generate_explanations(config_path, dataset_config_dir=args.dataset_config_dir)
        for kind, path in paths.items():
            print(f"Wrote {kind}: {path}")


if __name__ == "__main__":
    main()
