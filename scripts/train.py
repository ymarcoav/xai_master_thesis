"""Train a model from processed data."""

from __future__ import annotations

import argparse

from xai_thesis.pipeline import run_experiment, tune_hyperparameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Search for hyperparameters first and write them into the experiment's model_params before training.",
    )
    parser.add_argument(
        "--dataset-config-dir",
        default="configs/datasets",
        help="Directory containing dataset YAML configs.",
    )
    args = parser.parse_args()

    if args.tune:
        tuned_params = tune_hyperparameters(args.experiment, dataset_config_dir=args.dataset_config_dir)
        print(f"Tuned model_params: {tuned_params}")

    metrics = run_experiment(args.experiment, dataset_config_dir=args.dataset_config_dir)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
