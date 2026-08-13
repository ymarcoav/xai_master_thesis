"""Plot dataset diagnostics (missing values, train/test split balance, feature correlation)
for one or all datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_thesis.pipeline import (
    generate_feature_correlation,
    generate_missing_value_summary,
    generate_train_test_split_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        help="Dataset config name without .yaml. If omitted, all dataset configs are plotted.",
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

    for config_path in config_paths:
        print(f"Wrote {generate_missing_value_summary(config_path)}")
        print(f"Wrote {generate_train_test_split_comparison(config_path)}")
        print(f"Wrote {generate_feature_correlation(config_path)}")


if __name__ == "__main__":
    main()
