"""Generate global explainability artifacts (SHAP bar, PFI) for a trained model.

Both are computed on the same class-balanced test sample (total=500, random_state=42), whose
per-class sizes are logged. Global SHAP is skipped where the experiment config's
skip_global_shap flag says so (default: skipped for cic_ids2017 svm only); PFI always runs.

Local SHAP/LIME plots are interview stimuli only - see scripts/interview_stimuli.py.
"""

from __future__ import annotations

import argparse
import logging
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
    parser.add_argument(
        "--shap-chunk-size",
        type=int,
        default=None,
        help="Compute the global SHAP sample in checkpointed chunks of this many rows, so an "
        "interrupted run resumes. Default: one call. Only matters where global SHAP is not skipped.",
    )
    parser.add_argument(
        "--shap-n-jobs",
        type=int,
        default=1,
        help="Chunks to run in parallel (with --shap-chunk-size).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.experiment:
        config_paths = [Path(args.experiment)]
    else:
        config_paths = sorted(Path(args.config_dir).glob("*.yaml"))

    for config_path in config_paths:
        paths = generate_explanations(
            config_path,
            dataset_config_dir=args.dataset_config_dir,
            shap_chunk_size=args.shap_chunk_size,
            shap_n_jobs=args.shap_n_jobs,
        )
        for kind, path in paths.items():
            print(f"Wrote {kind}: {path}")


if __name__ == "__main__":
    main()
