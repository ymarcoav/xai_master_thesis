"""Tables behind the thesis's global/local explanation results and disagreement analysis.

Everything here is read from caches the pipeline already wrote - nothing is recomputed:

  global  outputs/metrics/agreement/global/<experiment>__balanced500.npz - mean |SHAP| pooled
          across classes and the permutation importance (macro-F1 drop) behind each model's
          shap_bar.png / permutation_importance.png, on the 500-row class-balanced test sample.
  local   outputs/metrics/agreement/attributions/<experiment>.npz - the signed per-instance SHAP
          and LIME vectors of the agreement instance set (60 UNSW-NB15 / 56 CIC-IDS2017 rows,
          identical for every model), plus outputs/metrics/stimuli/<dataset>_error_profile.csv
          for each instance's true and predicted class.

All comparisons run on the raw (pre-encoding) features: SHAP is computed on the one-hot encoded
columns and summed back onto proto/service/state, while LIME and permutation importance already
work on raw columns - 40 features for UNSW-NB15, 59 for CIC-IDS2017.

Ranking follows src/xai_thesis/agreement.py exactly (np.argsort of -|value|), so every top-k set
here is the one the agreement heatmaps score. Ranks are 1-based positions in that ordering. A
feature whose value is zero (or, for a global importance, at most zero) has no meaningful position
among the other zeros, so its "rank in the other method" is left blank rather than reported as an
arbitrary number.

The sign comparison is local only: mean |SHAP| and permutation importance are both unsigned.
Locally, both methods explain the same class per instance (the predict_proba argmax), so a positive
value pushes toward that class and a negative one against it. LIME runs with discretized features,
so its sign says whether the instance's quartile bin for that feature pushes toward the class,
while SHAP's is relative to the background sample - comparable in direction, not identical in
meaning. Sign agreement is Krishna et al.'s definition: the fraction of the top-k features that
are in both top-k sets *and* carry the same sign.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from xai_thesis.agreement import agreement_fraction

GLOBAL_METHODS = ("shap", "permutation")
LOCAL_METHODS = ("shap", "lime")
REPORT_K_VALUES = (5, 10, 20)
TOP_N = 10


def rank_order(values: np.ndarray) -> np.ndarray:
    """Feature indices from most to least important by |value| - agreement_fraction's own order."""
    return np.argsort(-np.abs(np.asarray(values, dtype=float))[None, :], axis=1)[0]


def rank_positions(values: np.ndarray) -> np.ndarray:
    """1-based rank of every feature under rank_order, as a [n_features] array."""
    positions = np.empty(len(values), dtype=int)
    positions[rank_order(values)] = np.arange(1, len(values) + 1)
    return positions


def top_n_rows(values: np.ndarray, feature_names: Sequence[str], n: int = TOP_N) -> list[dict]:
    """The n highest-|value| features as {rank, feature, value} rows, most important first."""
    return [
        {"rank": rank, "feature": feature_names[index], "value": float(values[index])}
        for rank, index in enumerate(rank_order(values)[:n], start=1)
    ]


def overlap_scores(values_a: np.ndarray, values_b: np.ndarray, k: int) -> dict[str, float]:
    """Feature and rank agreement at k between two single attribution/importance vectors."""
    a, b = np.asarray(values_a, dtype=float)[None, :], np.asarray(values_b, dtype=float)[None, :]
    return {
        "feature_agreement": float(agreement_fraction(a, b, k, "feature")[0]),
        "rank_agreement": float(agreement_fraction(a, b, k, "rank")[0]),
    }


def sign_agreement(values_a: np.ndarray, values_b: np.ndarray, k: int) -> float:
    """Krishna et al.'s sign agreement: share of the top-k in both sets with the same sign."""
    top_a, top_b = set(rank_order(values_a)[:k]), set(rank_order(values_b)[:k])
    same_sign = [i for i in top_a & top_b if np.sign(values_a[i]) == np.sign(values_b[i]) != 0]
    return len(same_sign) / k


def top_n_disagreements(
    values_a: np.ndarray,
    values_b: np.ndarray,
    feature_names: Sequence[str],
    method_a: str,
    method_b: str,
    *,
    n: int = TOP_N,
    signed: bool,
) -> list[dict]:
    """Features in one method's top n but not the other's, plus (signed only) shared-top-n sign flips.

    Each row names the feature, the method whose top n it is in, its rank and value there, and its
    rank and value under the other method. That second rank is what separates a strong disagreement
    (rank 2 in one method, rank 30 in the other) from a borderline one (rank 9 vs rank 11). It is
    blank when the other method gives the feature no importance at all (zero, or for an unsigned
    global importance, at most zero), since a position among tied zeros is arbitrary.
    """
    values = {method_a: np.asarray(values_a, dtype=float), method_b: np.asarray(values_b, dtype=float)}
    positions = {method: rank_positions(vector) for method, vector in values.items()}
    tops = {method: set(rank_order(vector)[:n]) for method, vector in values.items()}

    def has_importance(method: str, index: int) -> bool:
        value = values[method][index]
        return value != 0 if signed else value > 0

    rows: list[dict] = []
    for method, other in ((method_a, method_b), (method_b, method_a)):
        for index in sorted(tops[method] - tops[other], key=lambda i: positions[method][i]):
            rows.append(
                {
                    "kind": f"only_in_{method}_top{n}",
                    "feature": feature_names[index],
                    f"{method}_rank": int(positions[method][index]),
                    f"{method}_value": float(values[method][index]),
                    f"{other}_rank": int(positions[other][index]) if has_importance(other, index) else None,
                    f"{other}_value": float(values[other][index]),
                }
            )
    if signed:
        shared = sorted(tops[method_a] & tops[method_b], key=lambda i: positions[method_a][i])
        for index in shared:
            if np.sign(values[method_a][index]) * np.sign(values[method_b][index]) < 0:
                rows.append(
                    {
                        "kind": "sign_disagreement",
                        "feature": feature_names[index],
                        f"{method_a}_rank": int(positions[method_a][index]),
                        f"{method_a}_value": float(values[method_a][index]),
                        f"{method_b}_rank": int(positions[method_b][index]),
                        f"{method_b}_value": float(values[method_b][index]),
                    }
                )
    return rows


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6g")
    return path


def build_global_report(
    dataset_name: str,
    models: Sequence[str],
    *,
    metrics_dir: str | Path = "outputs/metrics",
    output_dir: str | Path = "outputs/metrics/explanation_report",
    sample_tag: str = "balanced500",
    n: int = TOP_N,
    k_values: Sequence[int] = REPORT_K_VALUES,
) -> dict[str, Path]:
    """Global top-n tables, method and model-pair overlap, and top-n disagreements for one dataset.

    Models without a cached global ranking are skipped and listed in the overlap CSV's
    "missing_models" note (none at present: every model/dataset pair has one). Model pairs are
    every pair of the remaining models (no reference model), per method.
    """
    cache_dir = Path(metrics_dir) / "agreement" / "global"
    rankings: dict[str, dict[str, np.ndarray]] = {}
    feature_names: list[str] | None = None
    for model in models:
        cache_path = cache_dir / f"{dataset_name}_{model}_baseline__{sample_tag}.npz"
        if not cache_path.exists():
            continue
        cached = np.load(cache_path, allow_pickle=False)
        names = [str(name) for name in cached["feature_names"]]
        if feature_names is not None and names != feature_names:
            raise ValueError(f"{cache_path.name}: feature order differs from the other models")
        feature_names = names
        rankings[model] = {method: cached[method].astype(float) for method in GLOBAL_METHODS}
    if feature_names is None:
        raise FileNotFoundError(f"no global ranking caches for {dataset_name} in {cache_dir}")
    missing = [model for model in models if model not in rankings]

    # Ranked on the importance clipped at zero, exactly as generate_global_agreement does; the
    # reported value is the raw one, so a negative permutation importance stays visible.
    clipped = {
        model: {method: np.clip(vector, 0.0, None) for method, vector in by_method.items()}
        for model, by_method in rankings.items()
    }

    top_rows = []
    for model, by_method in rankings.items():
        for method in GLOBAL_METHODS:
            for row in top_n_rows(clipped[model][method], feature_names, n):
                row["value"] = float(by_method[method][feature_names.index(row["feature"])])
                top_rows.append({"dataset": dataset_name, "model": model, "method": method, **row})

    overlap_rows = []
    for model in rankings:
        for k in k_values:
            overlap_rows.append(
                {
                    "dataset": dataset_name,
                    "comparison": "methods",
                    "method": "shap_vs_permutation",
                    "model_a": model,
                    "model_b": model,
                    "k": k,
                    **overlap_scores(clipped[model]["shap"], clipped[model]["permutation"], k),
                }
            )
    for method in GLOBAL_METHODS:
        for model_a, model_b in itertools.combinations(rankings, 2):
            for k in k_values:
                overlap_rows.append(
                    {
                        "dataset": dataset_name,
                        "comparison": "models",
                        "method": method,
                        "model_a": model_a,
                        "model_b": model_b,
                        "k": k,
                        **overlap_scores(clipped[model_a][method], clipped[model_b][method], k),
                    }
                )
    overlap = pd.DataFrame(overlap_rows)
    overlap["missing_models"] = "; ".join(missing)

    disagreement_rows = []
    for model, by_method in clipped.items():
        for row in top_n_disagreements(
            by_method["shap"], by_method["permutation"], feature_names, "shap", "permutation", n=n, signed=False
        ):
            # Report raw (unclipped) values next to the ranks computed on the clipped ones.
            index = feature_names.index(row["feature"])
            row["shap_value"] = float(rankings[model]["shap"][index])
            row["permutation_value"] = float(rankings[model]["permutation"][index])
            disagreement_rows.append({"dataset": dataset_name, "model": model, **row})

    output_dir = Path(output_dir)
    columns = ["dataset", "model", "kind", "feature", "shap_rank", "shap_value", "permutation_rank",
               "permutation_value"]
    return {
        "global_top": _write(pd.DataFrame(top_rows), output_dir / f"{dataset_name}_global_top{n}.csv"),
        "global_overlap": _write(overlap, output_dir / f"{dataset_name}_global_overlap.csv"),
        "global_disagreement": _write(
            pd.DataFrame(disagreement_rows, columns=columns),
            output_dir / f"{dataset_name}_global_disagreement.csv",
        ),
    }


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

    Writes the instance manifest with per-instance overlap, the signed top-n features of both
    methods per instance, the per-instance disagreements (top-n membership and sign), and a
    per-model summary split by all / correct / misclassified instances.
    """
    metrics_dir = Path(metrics_dir)
    profile = pd.read_csv(metrics_dir / "stimuli" / f"{dataset_name}_error_profile.csv")

    instance_rows, top_rows, disagreement_rows = [], [], []
    for model in models:
        cached = np.load(metrics_dir / "agreement" / "attributions" / f"{dataset_name}_{model}_baseline.npz")
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
            disagreements = top_n_disagreements(
                shap_values, lime_values, feature_names, "shap", "lime", n=n, signed=True
            )
            n_sign_flips = sum(item["kind"] == "sign_disagreement" for item in disagreements)
            shared_top = len(set(rank_order(shap_values)[:n]) & set(rank_order(lime_values)[:n]))

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
                "top1_match": rank_order(shap_values)[0] == rank_order(lime_values)[0],
            }
            for k in k_values:
                scores = overlap_scores(shap_values, lime_values, k)
                record[f"feature_agreement@{k}"] = scores["feature_agreement"]
                record[f"rank_agreement@{k}"] = scores["rank_agreement"]
                record[f"sign_agreement@{k}"] = sign_agreement(shap_values, lime_values, k)
            record[f"shared_top{n}"] = shared_top
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
            summary["top1_match_rate"] = rows["top1_match"].mean()
            for k in k_values:
                for metric in ("feature", "rank", "sign"):
                    column = f"{metric}_agreement@{k}"
                    summary[f"{column}_mean"] = rows[column].mean()
                    summary[f"{column}_sem"] = rows[column].sem() if len(rows) > 1 else float("nan")
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
