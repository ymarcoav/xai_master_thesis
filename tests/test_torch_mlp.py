import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import recall_score

from xai_thesis.torch_mlp import TorchMLPClassifier, _build_mlp


def _imbalanced_dataset(random_state: int = 0):
    """Three classes; 'rare' is a heavy minority overlapping class 'a', so an unweighted fit
    has a real incentive to ignore it."""
    rng = np.random.default_rng(random_state)
    x = np.vstack(
        [
            rng.normal(0.0, 1.0, (400, 6)),
            rng.normal(4.0, 1.0, (400, 6)),
            rng.normal(0.8, 1.0, (20, 6)),
        ]
    ).astype(np.float64)
    y = np.array(["a"] * 400 + ["b"] * 400 + ["rare"] * 20)
    return x, y


def test_build_mlp_has_one_relu_per_hidden_layer_and_linear_logit_head() -> None:
    net = _build_mlp(in_features=5, hidden_sizes=(8, 4), n_classes=3)

    kinds = [type(layer).__name__ for layer in net]
    assert kinds == ["Linear", "ReLU", "Linear", "ReLU", "Linear"]
    assert net[0].in_features == 5
    assert net[-1].out_features == 3  # raw logits, no softmax module


def test_fit_predict_round_trips_original_string_labels() -> None:
    x, y = _imbalanced_dataset()
    model = TorchMLPClassifier(hidden_layer_sizes=(32,), max_iter=40, random_state=42).fit(x, y)

    assert list(model.classes_) == ["a", "b", "rare"]
    predictions = model.predict(x)
    assert set(predictions) <= {"a", "b", "rare"}
    assert predictions.shape == (len(y),)


def test_predict_proba_is_softmax_over_classes() -> None:
    x, y = _imbalanced_dataset()
    model = TorchMLPClassifier(hidden_layer_sizes=(32,), max_iter=40, random_state=42).fit(x, y)

    proba = model.predict_proba(x)

    assert proba.shape == (len(y), 3)
    assert proba.dtype == np.float64
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (model.classes_[proba.argmax(axis=1)] == model.predict(x)).all()


def test_is_a_cloneable_sklearn_estimator() -> None:
    model = TorchMLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate_init=0.05)

    fresh = clone(model)

    assert fresh.get_params()["hidden_layer_sizes"] == (64, 32)
    assert fresh.get_params()["alpha"] == 0.01
    assert fresh.get_params()["learning_rate_init"] == 0.05
    assert not hasattr(fresh, "_model")  # clone is unfitted


def test_same_random_state_gives_identical_probabilities() -> None:
    x, y = _imbalanced_dataset()
    kwargs = dict(hidden_layer_sizes=(32, 16), max_iter=50, random_state=7, batch_size=128)

    first = TorchMLPClassifier(**kwargs).fit(x, y).predict_proba(x)
    second = TorchMLPClassifier(**kwargs).fit(x, y).predict_proba(x)

    assert np.array_equal(first, second)


def test_balanced_weights_improve_rare_class_recall_over_unweighted() -> None:
    # Checks the class-weight indexing: w_c must belong to the LabelEncoder's integer code c.
    # With misindexed weights, balancing would not help the rare class.
    x, y = _imbalanced_dataset()
    common = dict(hidden_layer_sizes=(64, 32), max_iter=80, random_state=1, batch_size=128)

    unweighted = TorchMLPClassifier(class_weight=None, **common).fit(x, y)
    balanced = TorchMLPClassifier(class_weight="balanced", **common).fit(x, y)

    recall_unweighted = recall_score(y, unweighted.predict(x), labels=["rare"], average="macro", zero_division=0)
    recall_balanced = recall_score(y, balanced.predict(x), labels=["rare"], average="macro", zero_division=0)

    assert recall_balanced > recall_unweighted
    assert recall_balanced >= 0.5


def test_early_stopping_stops_before_max_iter_on_a_trivially_separable_problem() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-5, 0.2, (200, 4)), rng.normal(5, 0.2, (200, 4))]).astype(np.float64)
    y = np.array(["low"] * 200 + ["high"] * 200)

    model = TorchMLPClassifier(
        hidden_layer_sizes=(16,), max_iter=500, early_stopping=True, n_iter_no_change=5, random_state=0
    ).fit(x, y)

    assert model.n_iter_ < 500


def test_rejects_unknown_class_weight() -> None:
    x, y = _imbalanced_dataset()
    with pytest.raises(ValueError, match="class_weight"):
        TorchMLPClassifier(class_weight="uniform").fit(x, y)
