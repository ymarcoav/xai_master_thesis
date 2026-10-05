"""Clean and split one or all datasets into model-ready train/test files."""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_thesis.pipeline import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        help="Dataset config name without .yaml. If omitted, all dataset configs are processed.",
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
        train_path, test_path = preprocess_dataset(config_path)
        print(f"Wrote {train_path} and {test_path}")


if __name__ == "__main__":
    main()
