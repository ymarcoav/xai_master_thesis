"""Train a model from processed data."""

from __future__ import annotations

import argparse

from xai_thesis.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()

    metrics = run_experiment(args.experiment)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
