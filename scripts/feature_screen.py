"""Rerun the constant / near-duplicate feature screen and check it against each dataset config.

Cleans each dataset as for training but keeps its screen_drop_columns, screens every numeric
feature (src/xai_thesis/screening.py), and writes the evidence - constant columns, and every
pair with |r| >= 0.99 together with the number of rows whose values differ - to
outputs/metrics/dataset_inspection/<dataset>/feature_screen.csv. Exits with status 1 if the
screen disagrees with a config's screen_drop_columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xai_thesis.pipeline import generate_feature_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        help="Dataset config name without .yaml. If omitted, all dataset configs are screened.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/datasets",
        help="Directory containing dataset YAML configs.",
    )
    args = parser.parse_args()

    if args.dataset:
        config_paths = [Path(args.config_dir) / f"{args.dataset}.yaml"]
    else:
        config_paths = sorted(Path(args.config_dir).glob("*.yaml"))

    consistent = True
    for config_path in config_paths:
        csv_path, problems = generate_feature_screen(config_path)
        print(f"Wrote {csv_path}")
        for problem in problems:
            print(f"  MISMATCH ({config_path.stem}): {problem}")
        if problems:
            consistent = False
        else:
            print(f"  {config_path.stem}: screen matches screen_drop_columns")
    if not consistent:
        sys.exit(1)


if __name__ == "__main__":
    main()
