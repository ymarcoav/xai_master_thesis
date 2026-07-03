"""Inspect raw dataset files and write summary reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_thesis.inspection import inspect_all_dataset_configs, inspect_dataset_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        help="Dataset config name without .yaml. If omitted, all dataset configs are inspected.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/datasets",
        help="Directory containing dataset YAML configs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/metrics/dataset_inspection",
        help="Directory where inspection reports are written.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum rows to scan per CSV file. By default all rows are scanned.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows to read per chunk while scanning CSV files.",
    )
    args = parser.parse_args()

    if args.dataset:
        config_path = Path(args.config_dir) / f"{args.dataset}.yaml"
        summaries = [
            inspect_dataset_config(
                config_path,
                args.output_dir,
                max_rows=args.max_rows,
                chunksize=args.chunksize,
            )
        ]
    else:
        summaries = inspect_all_dataset_configs(
            args.config_dir,
            args.output_dir,
            max_rows=args.max_rows,
            chunksize=args.chunksize,
        )

    for summary in summaries:
        output_path = Path(args.output_dir) / summary["dataset"]
        print(f"Wrote {summary['dataset']} inspection to {output_path}")


if __name__ == "__main__":
    main()
