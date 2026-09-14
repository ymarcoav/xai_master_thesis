"""Error profiles for choosing interview stimuli.

A misclassification makes a strong interview stimulus when the analyst has to reason about the
explanation rather than dismiss it: the model is confident in its wrong answer, and the flow
carries real traffic in both directions, so the explanation cannot be read off trivially as
"there was no response". This module profiles instances on exactly those properties - error type,
the model's confidence in the class it output, and what each side of the flow carried - so the
stimulus choice rests on data rather than argument.

scripts/stimulus_candidates.py runs it over two scopes: the agreement instance set (SHAP and LIME
attributions already exist for every row there) and every benign flow in the test split (the only
way to tell whether confident false positives exist at all, since the instance set holds just a
handful of benign rows).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

BENIGN_LABEL = "Benign"

# Flow-level traffic columns each dataset records per direction. "Forward" is the initiator's
# side (UNSW-NB15's source, CIC-IDS2017's Fwd), "backward" the responder's.
TRAFFIC_COLUMNS: dict[str, dict[str, str]] = {
    "unsw_nb15": {
        "duration": "dur",
        "fwd_packets": "spkts",
        "bwd_packets": "dpkts",
        "fwd_bytes": "sbytes",
        "bwd_bytes": "dbytes",
    },
    "cic_ids2017": {
        "duration": "Flow Duration",
        "fwd_packets": "Total Fwd Packets",
        "bwd_packets": "Total Backward Packets",
        "fwd_bytes": "Total Length of Fwd Packets",
        "bwd_bytes": "Total Length of Bwd Packets",
    },
}

ERROR_TYPES = ("correct", "false_positive", "false_negative", "attack_confusion")
MISCLASSIFICATION_TYPES = ERROR_TYPES[1:]
BACKWARD_TRAFFIC_STATUSES = ("none", "packets_only", "payload")


def classify_error(true_class: str, predicted_class: str, benign_label: str = BENIGN_LABEL) -> str:
    """IDS error type for one prediction, treating benign traffic as the negative class.

    false_positive   benign flow the model alerted on (predicted some attack)
    false_negative   attack the model let through (predicted benign)
    attack_confusion attack the model detected but put in the wrong attack class
    """
    if predicted_class == true_class:
        return "correct"
    if true_class == benign_label:
        return "false_positive"
    if predicted_class == benign_label:
        return "false_negative"
    return "attack_confusion"


def backward_traffic_status(bwd_packets: float, bwd_bytes: float) -> str:
    """What the responder side of a flow carried.

    none         no backward packets at all - one-directional, e.g. an unanswered probe
    packets_only backward packets but no backward bytes recorded
    payload      backward bytes present - the responder sent *something*, however little
    """
    if bwd_packets <= 0:
        return "none"
    if bwd_bytes <= 0:
        return "packets_only"
    return "payload"


def profile_predictions(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    *,
    traffic_columns: Mapping[str, str],
    benign_label: str = BENIGN_LABEL,
) -> pd.DataFrame:
    """One row per instance: true and predicted class, error type, confidence, traffic.

    `confidence` is the probability the model assigned to the class it actually output via
    .predict() - the class an alert would show, and the class the SHAP waterfall stimuli explain
    (_generate_instance_explanations explains pipeline.predict()). `max_proba` and
    `proba_argmax_class` are reported alongside because they can differ from it: SVC's .predict()
    uses one-vs-one voting while predict_proba is Platt-scaled, so for svm the predicted class is
    not always the most probable one, and its confidence can be well below max_proba.

    traffic_columns maps duration / fwd_packets / bwd_packets / fwd_bytes / bwd_bytes onto the
    dataset's own column names (see TRAFFIC_COLUMNS).
    """
    classes = [str(label) for label in pipeline.named_steps["model"].classes_]
    predicted = np.asarray(pipeline.predict(x)).astype(str)
    proba = np.asarray(pipeline.predict_proba(x), dtype=float)
    true = y.loc[x.index].astype(str).to_numpy()

    class_position = {label: position for position, label in enumerate(classes)}
    predicted_index = np.array([class_position[label] for label in predicted])
    true_index = np.array([class_position[label] for label in true])
    rows = np.arange(len(x))

    profile = pd.DataFrame(
        {
            "instance_index": x.index.to_numpy(),
            "true_class": true,
            "predicted_class": predicted,
            "error_type": [classify_error(t, p, benign_label) for t, p in zip(true, predicted)],
            "confidence": proba[rows, predicted_index],
            "proba_true_class": proba[rows, true_index],
            "max_proba": proba.max(axis=1),
            "proba_argmax_class": np.array(classes)[proba.argmax(axis=1)],
        }
    )
    profile["predict_disagrees_with_proba"] = profile["predicted_class"] != profile["proba_argmax_class"]
    for key in ("duration", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes"):
        profile[key] = x[traffic_columns[key]].to_numpy()
    profile["backward_traffic"] = [
        backward_traffic_status(packets, size)
        for packets, size in zip(profile["bwd_packets"], profile["bwd_bytes"])
    ]
    return profile


def candidate_stimuli(
    profile: pd.DataFrame,
    *,
    min_confidence: float,
    error_types: Sequence[str] = MISCLASSIFICATION_TYPES,
    backward_traffic: Sequence[str] = ("payload",),
    min_fwd_bytes: float = 0.0,
    min_bwd_bytes: float = 0.0,
) -> pd.DataFrame:
    """Misclassifications the model was confident about, on flows that carried traffic.

    backward_traffic="payload" only asks for *some* responder bytes, which a 6-byte exchange
    passes, so min_fwd_bytes / min_bwd_bytes set a real per-direction floor for "substantive".
    Sorted most confident first. An empty result means no profiled error meets the bar - not that
    such errors cannot occur elsewhere in the test split.
    """
    keep = (
        profile["error_type"].isin(error_types)
        & (profile["confidence"] >= min_confidence)
        & profile["backward_traffic"].isin(backward_traffic)
        & (profile["fwd_bytes"] >= min_fwd_bytes)
        & (profile["bwd_bytes"] >= min_bwd_bytes)
    )
    return profile[keep].sort_values("confidence", ascending=False)
