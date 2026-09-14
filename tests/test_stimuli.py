import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from xai_thesis.models import build_model
from xai_thesis.preprocessing import build_preprocessing_pipeline
from xai_thesis.stimuli import (
    backward_traffic_status,
    candidate_stimuli,
    classify_error,
    profile_predictions,
)

TOY_TRAFFIC = {
    "duration": "dur",
    "fwd_packets": "fwd_p",
    "bwd_packets": "bwd_p",
    "fwd_bytes": "fwd_b",
    "bwd_bytes": "bwd_b",
}


@pytest.mark.parametrize(
    ("true", "predicted", "expected"),
    [
        ("Benign", "Benign", "correct"),
        ("DoS", "DoS", "correct"),
        ("Benign", "DoS", "false_positive"),
        ("DoS", "Benign", "false_negative"),
        ("DoS", "Scan", "attack_confusion"),
    ],
)
def test_classify_error_covers_every_error_type(true: str, predicted: str, expected: str) -> None:
    assert classify_error(true, predicted) == expected


def test_backward_traffic_status_distinguishes_none_packets_and_payload() -> None:
    assert backward_traffic_status(0, 0) == "none"
    assert backward_traffic_status(3, 0) == "packets_only"
    assert backward_traffic_status(3, 180) == "payload"


def _toy_pipeline_and_data() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 90
    y = pd.Series(np.repeat(["Benign", "DoS", "Scan"], n // 3))
    x = pd.DataFrame(
        {
            "dur": rng.exponential(1.0, n),
            "fwd_p": rng.integers(1, 20, n),
            "bwd_p": rng.integers(0, 20, n),
            "fwd_b": rng.integers(40, 5000, n),
            "bwd_b": rng.integers(0, 5000, n),
            "signal": np.repeat([0.0, 3.0, 6.0], n // 3) + rng.normal(0, 2.0, n),
        }
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline(x, categorical_columns=[])),
            ("model", build_model("random_forest", random_state=42, n_estimators=10)),
        ]
    )
    pipeline.fit(x, y)
    return pipeline, x, y


def test_profile_predictions_reports_confidence_in_the_class_the_model_output() -> None:
    pipeline, x, y = _toy_pipeline_and_data()

    profile = profile_predictions(pipeline, x, y, traffic_columns=TOY_TRAFFIC)

    assert len(profile) == len(x)
    assert list(profile["instance_index"]) == list(x.index)
    # confidence is the probability of the class .predict() returned, not just the max.
    proba = pipeline.predict_proba(x)
    classes = list(pipeline.named_steps["model"].classes_)
    expected = [proba[i, classes.index(label)] for i, label in enumerate(pipeline.predict(x))]
    assert np.allclose(profile["confidence"], expected)
    assert ((profile["confidence"] >= 0) & (profile["confidence"] <= 1)).all()
    # error_type is consistent with the predicted/true classes on every row.
    assert ((profile["error_type"] == "correct") == (profile["true_class"] == profile["predicted_class"])).all()
    # traffic is carried through from the raw columns, and classified per row.
    assert profile["bwd_packets"].tolist() == x["bwd_p"].tolist()
    assert set(profile["backward_traffic"]) <= {"none", "packets_only", "payload"}


def test_candidate_stimuli_keeps_only_confident_errors_with_responder_payload() -> None:
    profile = pd.DataFrame(
        {
            "instance_index": [1, 2, 3, 4, 5, 6],
            "error_type": ["false_positive", "false_positive", "correct", "false_negative", "attack_confusion", "false_positive"],
            "confidence": [0.95, 0.60, 0.99, 0.92, 0.97, 0.99],
            "backward_traffic": ["payload", "payload", "payload", "payload", "payload", "none"],
            "fwd_bytes": [900, 900, 900, 900, 900, 900],
            "bwd_bytes": [500, 500, 500, 500, 500, 0],
        }
    )

    candidates = candidate_stimuli(profile, min_confidence=0.9)

    # 2 is not confident, 3 is correct, 6 has no responder traffic; the rest sorted by confidence.
    assert candidates["instance_index"].tolist() == [5, 1, 4]
    assert candidate_stimuli(profile, min_confidence=0.999).empty


def test_candidate_stimuli_byte_floor_rejects_near_empty_flows() -> None:
    # A 6-byte exchange counts as responder "payload" but is not a substantive flow - the
    # CIC-IDS2017 false positives in the instance set look exactly like this.
    profile = pd.DataFrame(
        {
            "instance_index": [1, 2, 3],
            "error_type": ["false_positive"] * 3,
            "confidence": [0.99, 0.98, 0.97],
            "backward_traffic": ["payload"] * 3,
            "fwd_bytes": [6, 1200, 1200],
            "bwd_bytes": [6, 460, 40],
        }
    )

    assert candidate_stimuli(profile, min_confidence=0.9)["instance_index"].tolist() == [1, 2, 3]
    floored = candidate_stimuli(profile, min_confidence=0.9, min_fwd_bytes=100, min_bwd_bytes=100)
    assert floored["instance_index"].tolist() == [2]
