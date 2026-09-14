"""Profile every model's errors to choose interview stimuli.

Two scopes:

  instance_set  The class-balanced instance set the local SHAP-vs-LIME panel is computed on, so
                attributions already exist for every row. Writes
                outputs/metrics/stimuli/<dataset>_error_profile.csv (every instance, every model).

  benign        Every benign flow in the test split, keeping the false positives - benign traffic
                the model alerted on. The instance set holds only 6 (UNSW-NB15) / 8 (CIC-IDS2017)
                benign rows, far too few to say whether confident false positives exist; this scope
                is how that is actually answered. Writes
                outputs/metrics/stimuli/<dataset>_benign_false_positives.csv.

For each model it prints the confidence distribution and backward-traffic status of the errors,
and the candidates: errors the model was confident about, on flows that carried at least
--min-bytes in each direction. Each model's rows are written as soon as it is profiled, and a
rerun replaces only the models it covers, so a slow model (svm) can be run on its own.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xai_thesis.pipeline import (
    _AGREEMENT_SAMPLE_SIZE,
    _class_balanced_sample,
    _dataset_experiment_paths,
    _load_experiment_and_data,
)
from xai_thesis.stimuli import (
    BACKWARD_TRAFFIC_STATUSES,
    BENIGN_LABEL,
    MISCLASSIFICATION_TYPES,
    TRAFFIC_COLUMNS,
    candidate_stimuli,
    profile_predictions,
)
from xai_thesis.utils import ensure_directory, load_yaml

_BYTE_BUCKETS = ("0", "1-99", "100-999", ">=1000")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(TRAFFIC_COLUMNS), help="Default: both datasets.")
    parser.add_argument(
        "--scope",
        choices=("instance_set", "benign"),
        default="instance_set",
        help="Which rows to profile (default: instance_set).",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Restrict to these models (default: all).")
    parser.add_argument("--config-dir", default="configs/experiments")
    parser.add_argument("--dataset-config-dir", default="configs/datasets")
    parser.add_argument("--models-dir", default="outputs/models")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument(
        "--n-instances",
        type=int,
        default=_AGREEMENT_SAMPLE_SIZE,
        help="Size of the class-balanced instance set (default: the agreement panel's own).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.9,
        help="Confidence bar for candidates (default: 0.9). The full distribution is printed "
        "regardless, so the effect of moving the bar is visible.",
    )
    parser.add_argument(
        "--min-bytes",
        type=float,
        default=100.0,
        help="Bytes each direction of a candidate flow must carry (default: 100). Responder "
        "'payload' alone needs only one byte, which a 6-byte exchange passes.",
    )
    parser.add_argument("--top", type=int, default=10, help="Candidates to print per model (all are in the CSV).")
    args = parser.parse_args()

    for dataset_name in [args.dataset] if args.dataset else sorted(TRAFFIC_COLUMNS):
        experiments = _dataset_experiment_paths(dataset_name, args.config_dir)
        if args.models:
            experiments = [(model, path) for model, path in experiments if model in args.models]
        if not experiments:
            raise ValueError(f"no experiments for {dataset_name} match --models {args.models}")

        # Data once per dataset.
        _, _, _, _, x_test, y_test = _load_experiment_and_data(experiments[0][1], args.dataset_config_dir)
        if args.scope == "instance_set":
            seeds = {load_yaml(path).get("random_state", 42) for _, path in experiments}
            if len(seeds) != 1:
                raise ValueError(f"{dataset_name} experiments disagree on random_state: {seeds}")
            rows = _class_balanced_sample(x_test, y_test, args.n_instances, random_state=seeds.pop())
            name, header = "error_profile", f"{len(rows)} instances, class mix {dict(y_test.loc[rows.index].value_counts())}"
        else:
            rows = x_test.loc[y_test.astype(str) == BENIGN_LABEL]
            name, header = "benign_false_positives", f"{len(rows)} benign test flows"
        rows_y = y_test.loc[rows.index]
        out_path = ensure_directory(Path(args.metrics_dir) / "stimuli") / f"{dataset_name}_{name}.csv"

        print(f"\n{'=' * 100}\n{dataset_name}: {header}\nwriting {out_path}\n{'=' * 100}", flush=True)
        for model_name, config_path in experiments:
            started = time.time()
            pipeline = joblib.load(Path(args.models_dir) / f"{load_yaml(config_path)['name']}.joblib")
            profile = profile_predictions(pipeline, rows, rows_y, traffic_columns=TRAFFIC_COLUMNS[dataset_name])
            if args.scope == "benign":
                profile = profile[profile["error_type"] == "false_positive"]
            profile.insert(0, "model", model_name)
            _upsert(out_path, model_name, profile)

            if args.scope == "instance_set":
                errors = profile[profile["error_type"].isin(MISCLASSIFICATION_TYPES)]
                counts = profile["error_type"].value_counts()
                print(f"\n--- {model_name}: {len(errors)}/{len(profile)} misclassified "
                      f"(FP {counts.get('false_positive', 0)}, FN {counts.get('false_negative', 0)}, "
                      f"attack confusion {counts.get('attack_confusion', 0)})   [{time.time() - started:.0f}s]")
            else:
                errors = profile
                print(f"\n--- {model_name}: {len(errors)} false positives in {len(rows)} benign test flows "
                      f"({100 * len(errors) / len(rows):.3f}%)   [{time.time() - started:.0f}s]")
            _describe(errors, min_confidence=args.min_confidence, min_bytes=args.min_bytes, top=args.top)


def _upsert(out_path: Path, model_name: str, profile: pd.DataFrame) -> None:
    """Replace this model's rows in the dataset's CSV, keeping every other model's."""
    if out_path.exists():
        existing = pd.read_csv(out_path)
        profile = pd.concat([existing[existing["model"] != model_name], profile], ignore_index=True)
    profile.to_csv(out_path, index=False)


def _describe(errors: pd.DataFrame, *, min_confidence: float, min_bytes: float, top: int) -> None:
    if errors.empty:
        return
    confidence = errors["confidence"]
    print(f"    confidence: min {confidence.min():.2f}  median {confidence.median():.2f}  "
          f"max {confidence.max():.2f}   >=0.8: {(confidence >= 0.8).sum()}   >=0.9: {(confidence >= 0.9).sum()}")
    status = errors["backward_traffic"].value_counts()
    print("    backward traffic: " + ", ".join(f"{s} {status.get(s, 0)}" for s in BACKWARD_TRAFFIC_STATUSES))
    disagree = int(errors["predict_disagrees_with_proba"].sum())
    if disagree:
        print(f"    note: .predict() disagrees with predict_proba argmax on {disagree} of these")

    confident = errors[confidence >= min_confidence]
    if not confident.empty:
        thinner = confident[["fwd_bytes", "bwd_bytes"]].min(axis=1)
        buckets = pd.cut(thinner, [-np.inf, 0, 99, 999, np.inf], labels=list(_BYTE_BUCKETS))
        spread = buckets.value_counts().reindex(list(_BYTE_BUCKETS), fill_value=0)
        print(f"    confident (>= {min_confidence}), by bytes in the thinner direction: "
              + ", ".join(f"{label}: {count}" for label, count in spread.items()))
        print("    confident, predicted as: "
              + ", ".join(f"{label} {count}" for label, count in confident["predicted_class"].value_counts().items()))

    candidates = candidate_stimuli(
        errors, min_confidence=min_confidence, min_fwd_bytes=min_bytes, min_bwd_bytes=min_bytes
    )
    shown = f", top {top} shown" if len(candidates) > top else ""
    print(f"    candidates (confidence >= {min_confidence}, >= {min_bytes:g} bytes each way): {len(candidates)}{shown}")
    for row in candidates.head(top).itertuples():
        print(f"      #{row.instance_index:<8} {row.true_class:>15} -> {row.predicted_class:<15} "
              f"conf {row.confidence:.3f}  {row.error_type:<16} "
              f"fwd {row.fwd_packets:g}p/{row.fwd_bytes:g}B  bwd {row.bwd_packets:g}p/{row.bwd_bytes:g}B  "
              f"dur {row.duration:g}")


if __name__ == "__main__":
    main()
