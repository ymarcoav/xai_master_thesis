"""Plot a confusion matrix for one or all trained experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_thesis.pipeline import generate_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        help="Path to an experiment YAML config. If omitted, all configs in --config-dir are plotted.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/experiments",
        help="Directory containing experiment YAML configs.",
    )
    args = parser.parse_args()

    if args.experiment:
        config_paths = [Path(args.experiment)]
    else:
        config_paths = sorted(Path(args.config_dir).glob("*.yaml"))

    for config_path in config_paths:
        output_path = generate_confusion_matrix(config_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
