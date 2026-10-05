"""Tables behind the explanation results reported in the thesis.

Everything is read from caches the pipeline has already written; nothing is recomputed:

  global  outputs/metrics/global_importance/<experiment>__{shap,pfi}.npz - global SHAP importance
          (mean |SHAP| over instances and classes) and permutation feature importance (macro-F1
          drop, 10 repeats), both on the same class-balanced test sample.
  local   outputs/metrics/agreement/attributions/<experiment>.npz - the signed per-instance SHAP
          and LIME attributions of the agreement instance set (60 UNSW-NB15 / 56 CIC-IDS2017
          instances, the same for every model), plus
          outputs/metrics/stimuli/<dataset>_error_profile.csv for each instance's true and
          predicted class.

The global table lists each method's top 10 per model; the only global comparison is the number
of features shared by the SHAP and permutation importance top 10s (global_overlap_table). The
local comparison is SHAP vs LIME feature and rank agreement (agreement.py).

All tables use the raw (pre-encoding) features: 40 for UNSW-NB15, 59 for CIC-IDS2017.

Ordering. Local attributions are ranked by |value|, ties broken by feature name
(agreement.ranked_feature_indices), so every top-k set here is the one the agreement metrics
score. Global importances are ranked by value, descending, ties by feature name; a negative
permutation importance means no measurable importance and ranks below zero. Ranks are 1-based.

Sign (local only). Both methods explain the same class per instance (the most probable class),
so a positive value pushes toward that class. LIME uses discretized features, so its sign refers
to the instance's quartile bin, while SHAP's is relative to the background sample. Sign agreement
(Krishna et al.) is the share of the top-k features that are in both top-k sets and carry the
same sign; it is a supplementary column, not one of the two agreement metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from xai_thesis.agreement import agreement_fraction, ranked_feature_indices, summarize_agreement

logger = logging.getLogger(__name__)

GLOBAL_METHODS = {"shap": "SHAP", "pfi": "PFI"}
REPORT_K_VALUES = (1, 5, 10)
TOP_N = 10


def rank_order(values: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Feature indices from most to least important by |value|, ties by feature name."""
    return ranked_feature_indices(np.asarray(values, dtype=float), feature_names)


def rank_positions(values: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """1-based rank of every feature under rank_order, as a [n_features] array."""
    positions = np.empty(len(values), dtype=int)
    positions[rank_order(values, feature_names)] = np.arange(1, len(values) + 1)
    return positions


def top_n_rows(values: np.ndarray, feature_names: Sequence[str], n: int = TOP_N) -> list[dict]:
    """The n highest-|value| features as {rank, feature, value} rows, most important first."""
    return [
        {"rank": rank, "feature": feature_names[index], "value": float(values[index])}
        for rank, index in enumerate(rank_order(values, feature_names)[:n], start=1)
    ]


def global_rank_order(importance: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Feature indices by global importance, descending (signed, not |value|), ties by name."""
    importance = np.asarray(importance, dtype=float)
    name_order = np.empty(len(feature_names), dtype=int)
    name_order[np.argsort(np.asarray(feature_names, dtype=str), kind="stable")] = np.arange(len(feature_names))
    return np.lexsort((name_order, -importance))


def sign_agreement(values_a: np.ndarray, values_b: np.ndarray, k: int, feature_names: Sequence[str]) -> float:
    """Krishna et al.'s sign agreement: share of the top-k in both sets with the same sign."""
    top_a = set(rank_order(values_a, feature_names)[:k])
    top_b = set(rank_order(values_b, feature_names)[:k])
    same_sign = [i for i in top_a & top_b if np.sign(values_a[i]) == np.sign(values_b[i]) != 0]
    return len(same_sign) / k


def top_n_disagreements(
    shap_values: np.ndarray,
    lime_values: np.ndarray,
    feature_names: Sequence[str],
    *,
    n: int = TOP_N,
) -> list[dict]:
    """One instance's SHAP-vs-LIME top-n membership disagreements and shared-top-n sign flips.

    Each row names the feature, the method whose top n it is in, its rank and value there, and its
    rank and value under the other method. That second rank is what separates a strong disagreement
    (rank 2 in one method, rank 30 in the other) from a borderline one (rank 9 vs rank 11). It is
    blank when the other method gives the feature an attribution of exactly zero, since a position
    among tied zeros is set only by the name tie-break.
    """
    values = {"shap": np.asarray(shap_values, dtype=float), "lime": np.asarray(lime_values, dtype=float)}
    positions = {method: rank_positions(vector, feature_names) for method, vector in values.items()}
    tops = {method: set(rank_order(vector, feature_names)[:n]) for method, vector in values.items()}

    rows: list[dict] = []
    for method, other in (("shap", "lime"), ("lime", "shap")):
        for index in sorted(tops[method] - tops[other], key=lambda i: positions[method][i]):
            rows.append(
                {
                    "kind": f"only_in_{method}_top{n}",
                    "feature": feature_names[index],
                    f"{method}_rank": int(positions[method][index]),
                    f"{method}_value": float(values[method][index]),
                    f"{other}_rank": int(positions[other][index]) if values[other][index] != 0 else None,
                    f"{other}_value": float(values[other][index]),
                }
            )
    for index in sorted(tops["shap"] & tops["lime"], key=lambda i: positions["shap"][i]):
        if np.sign(values["shap"][index]) * np.sign(values["lime"][index]) < 0:
            rows.append(
                {
                    "kind": "sign_disagreement",
                    "feature": feature_names[index],
                    "shap_rank": int(positions["shap"][index]),
                    "shap_value": float(values["shap"][index]),
                    "lime_rank": int(positions["lime"][index]),
                    "lime_value": float(values["lime"][index]),
                }
            )
    return rows


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6g")
    return path


def _attribution_cache(metrics_dir: str | Path, dataset_name: str, model: str) -> Path:
    return Path(metrics_dir) / "agreement" / "attributions" / f"{dataset_name}_{model}_baseline.npz"


def local_agreement_table(
    datasets: Sequence[str],
    models: Sequence[str],
    *,
    metrics_dir: str | Path = "outputs/metrics",
    k_values: Sequence[int] = REPORT_K_VALUES,
) -> pd.DataFrame:
    """The thesis's local results: dataset, model, k, fa_mean, fa_se, ra_mean, ra_se, n.

    Mean and standard error (sample SD with ddof=1 / sqrt(n)) of per-instance feature and rank
    agreement between SHAP and LIME. Raises if two models of one dataset were scored on different
    instances.
    """
    rows = []
    for dataset_name in datasets:
        reference = None
        for model in models:
            cached = np.load(_attribution_cache(metrics_dir, dataset_name, model), allow_pickle=False)
            if reference is None:
                reference = cached["instance_index"]
            elif not np.array_equal(cached["instance_index"], reference):
                raise ValueError(f"{dataset_name}/{model}: instance set differs from the other models'")
            feature_names = [str(name) for name in cached["feature_names"]]
            rows.extend(
                {"dataset": dataset_name, "model": model, **summary}
                for summary in summarize_agreement(cached["shap"], cached["lime"], k_values, feature_names)
            )
    return pd.DataFrame(rows, columns=["dataset", "model", "k", "fa_mean", "fa_se", "ra_mean", "ra_se", "n"])


def global_top_table(
    datasets: Sequence[str],
    models: Sequence[str],
    *,
    metrics_dir: str | Path = "outputs/metrics",
    n: int = TOP_N,
) -> pd.DataFrame:
    """The thesis's global results: dataset, model, method, rank, feature, importance, n_instances.

    One block of n rows per (dataset, model, method). A missing cache - global SHAP that was
    skipped and never computed - is logged and left out rather than raising.
    """
    rows = []
    for dataset_name in datasets:
        for model in models:
            for method, label in GLOBAL_METHODS.items():
                cache_path = Path(metrics_dir) / "global_importance" / f"{dataset_name}_{model}_baseline__{method}.npz"
                if not cache_path.exists():
                    logger.warning("%s/%s: no global %s cache (%s) - left out", dataset_name, model, label, cache_path)
                    continue
                cached = np.load(cache_path, allow_pickle=False)
                feature_names = [str(name) for name in cached["feature_names"]]
                importance = cached["importance"].astype(float)
                for rank, index in enumerate(global_rank_order(importance, feature_names)[:n], start=1):
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "method": label,
                            "rank": rank,
                            "feature": feature_names[index],
                            "importance": float(importance[index]),
                            "n_instances": int(cached["class_counts"].sum()),
                        }
                    )
    return pd.DataFrame(rows)


def global_overlap_table(
    datasets: Sequence[str],
    models: Sequence[str],
    *,
    metrics_dir: str | Path = "outputs/metrics",
    k: int = TOP_N,
) -> pd.DataFrame:
    """Global SHAP vs PFI top-k overlap: dataset, model, k, n_shared, feature_agreement, shared.

    n_shared is the number of features in both top-k rankings (ordered as in global_top_table),
    and feature_agreement = n_shared / k, the FA_k of the local analysis applied to the two global
    rankings. A model missing either cache is logged and left out. Raises if the two caches were
    computed on different instances.
    """
    rows = []
    for dataset_name in datasets:
        for model in models:
            paths = {
                method: Path(metrics_dir) / "global_importance" / f"{dataset_name}_{model}_baseline__{method}.npz"
                for method in GLOBAL_METHODS
            }
            missing = [GLOBAL_METHODS[method] for method, path in paths.items() if not path.exists()]
            if missing:
                logger.warning("%s/%s: no global %s cache - overlap left out", dataset_name, model, "/".join(missing))
                continue
            cached = {method: np.load(path, allow_pickle=False) for method, path in paths.items()}
            if not np.array_equal(cached["shap"]["instance_index"], cached["pfi"]["instance_index"]):
                raise ValueError(f"{dataset_name}/{model}: global SHAP and PFI were computed on different instances")
            tops = {}
            for method, data in cached.items():
                feature_names = [str(name) for name in data["feature_names"]]
                order = global_rank_order(data["importance"].astype(float), feature_names)[:k]
                tops[method] = [feature_names[index] for index in order]
            shared = [name for name in tops["shap"] if name in tops["pfi"]]
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model,
                    "k": k,
                    "n_shared": len(shared),
                    "feature_agreement": len(shared) / k,
                    "shared": "; ".join(shared),
                }
            )
    return pd.DataFrame(rows, columns=["dataset", "model", "k", "n_shared", "feature_agreement", "shared"])


def build_local_report(
    dataset_name: str,
    models: Sequence[str],
    *,
    metrics_dir: str | Path = "outputs/metrics",
    output_dir: str | Path = "outputs/metrics/explanation_report",
    n: int = TOP_N,
    k_values: Sequence[int] = REPORT_K_VALUES,
) -> dict[str, Path]:
    """Per-instance SHAP-vs-LIME tables over the agreement instance set for one dataset.

    Writes the instance manifest with per-instance feature/rank (and supplementary sign)
    agreement, the signed top-n features of both methods per instance, the per-instance
    disagreements (top-n membership and sign), and a per-model summary over all instances and,
    as a breakdown, over correctly classified and misclassified ones.
    """
    metrics_dir = Path(metrics_dir)
    profile = pd.read_csv(metrics_dir / "stimuli" / f"{dataset_name}_error_profile.csv")

    instance_rows, top_rows, disagreement_rows = [], [], []
    for model in models:
        cached = np.load(_attribution_cache(metrics_dir, dataset_name, model), allow_pickle=False)
        shap_matrix, lime_matrix = cached["shap"].astype(float), cached["lime"].astype(float)
        feature_names = [str(name) for name in cached["feature_names"]]
        model_profile = profile[profile["model"] == model].set_index("instance_index")

        for row, instance_index in enumerate(cached["instance_index"].tolist()):
            if instance_index not in model_profile.index:
                raise KeyError(f"{dataset_name}/{model}: instance {instance_index} missing from the error profile")
            info = model_profile.loc[instance_index]
            explained_class = str(cached["explained_class"][row])
            if explained_class != info["proba_argmax_class"]:
                raise ValueError(
                    f"{dataset_name}/{model}: instance {instance_index} explained as {explained_class} "
                    f"but its predict_proba argmax is {info['proba_argmax_class']} - caches out of sync"
                )
            shap_values, lime_values = shap_matrix[row], lime_matrix[row]
            disagreements = top_n_disagreements(shap_values, lime_values, feature_names, n=n)
            n_sign_flips = sum(item["kind"] == "sign_disagreement" for item in disagreements)

            record = {
                "dataset": dataset_name,
                "model": model,
                "instance_index": instance_index,
                "true_class": info["true_class"],
                "predicted_class": info["predicted_class"],
                "explained_class": explained_class,
                "correct": info["true_class"] == info["predicted_class"],
                "error_type": info["error_type"],
                "predict_disagrees_with_proba": bool(info["predict_disagrees_with_proba"]),
            }
            for k in k_values:
                for metric in ("feature", "rank"):
                    record[f"{metric}_agreement@{k}"] = float(
                        agreement_fraction(shap_values, lime_values, k, metric, feature_names)[0]
                    )
                record[f"sign_agreement@{k}"] = sign_agreement(shap_values, lime_values, k, feature_names)
            record[f"sign_disagreements_top{n}"] = n_sign_flips
            instance_rows.append(record)

            for method, values in (("shap", shap_values), ("lime", lime_values)):
                for item in top_n_rows(values, feature_names, n):
                    top_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "instance_index": instance_index,
                            "explained_class": explained_class,
                            "method": method,
                            **item,
                            "direction": "toward" if item["value"] > 0 else "against" if item["value"] < 0 else "none",
                        }
                    )
            for item in disagreements:
                disagreement_rows.append(
                    {"dataset": dataset_name, "model": model, "instance_index": instance_index, **item}
                )

    instances = pd.DataFrame(instance_rows)
    summary_rows = []
    for model in models:
        model_rows = instances[instances["model"] == model]
        for subset, rows in (
            ("all", model_rows),
            ("correct", model_rows[model_rows["correct"]]),
            ("misclassified", model_rows[~model_rows["correct"]]),
        ):
            if rows.empty:
                continue
            summary = {"dataset": dataset_name, "model": model, "subset": subset, "n_instances": len(rows)}
            for k in k_values:
                for metric in ("feature", "rank", "sign"):
                    column = f"{metric}_agreement@{k}"
                    summary[f"{column}_mean"] = rows[column].mean()
                    summary[f"{column}_se"] = rows[column].sem(ddof=1) if len(rows) > 1 else float("nan")
            summary[f"instances_with_sign_disagreement_top{n}"] = int((rows[f"sign_disagreements_top{n}"] > 0).sum())
            summary[f"mean_sign_disagreements_top{n}"] = rows[f"sign_disagreements_top{n}"].mean()
            summary_rows.append(summary)

    output_dir = Path(output_dir)
    columns = ["dataset", "model", "instance_index", "kind", "feature", "shap_rank", "shap_value",
               "lime_rank", "lime_value"]
    return {
        "local_instances": _write(instances, output_dir / f"{dataset_name}_local_instances.csv"),
        "local_top": _write(pd.DataFrame(top_rows), output_dir / f"{dataset_name}_local_top{n}.csv"),
        "local_disagreement": _write(
            pd.DataFrame(disagreement_rows, columns=columns),
            output_dir / f"{dataset_name}_local_disagreement.csv",
        ),
        "local_summary": _write(pd.DataFrame(summary_rows), output_dir / f"{dataset_name}_local_summary.csv"),
    }


def write_thesis_tables(
    datasets: Sequence[str],
    models: Sequence[str],
    sample_sizes: pd.DataFrame,
    *,
    metrics_dir: str | Path = "outputs/metrics",
    output_dir: str | Path = "outputs/metrics/explanation_report",
) -> dict[str, Path]:
    """The four thesis CSVs, each covering every dataset: local_agreement.csv, global_top10.csv,
    global_overlap.csv (SHAP vs PFI top-10 overlap) and sample_sizes.csv (per-class counts of the
    local and global samples)."""
    output_dir = Path(output_dir)
    return {
        "local_agreement": _write(
            local_agreement_table(datasets, models, metrics_dir=metrics_dir), output_dir / "local_agreement.csv"
        ),
        "global_top": _write(
            global_top_table(datasets, models, metrics_dir=metrics_dir), output_dir / f"global_top{TOP_N}.csv"
        ),
        "global_overlap": _write(
            global_overlap_table(datasets, models, metrics_dir=metrics_dir), output_dir / "global_overlap.csv"
        ),
        "sample_sizes": _write(sample_sizes, output_dir / "sample_sizes.csv"),
    }
