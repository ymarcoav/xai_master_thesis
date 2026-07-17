import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from xai_thesis.models import build_model, train_model


def test_build_model_svm_is_balanced_and_seeded() -> None:
    model = build_model("svm", random_state=7)

    assert isinstance(model, LinearSVC)
    assert model.random_state == 7
    assert model.class_weight == "balanced"


def test_build_model_svm_forwards_kwargs() -> None:
    model = build_model("svm", random_state=7, C=10, max_iter=5000)

    assert model.C == 10
    assert model.max_iter == 5000


def test_build_model_neural_network_has_no_class_weight() -> None:
    model = build_model("neural_network", random_state=7)

    assert isinstance(model, MLPClassifier)
    assert model.random_state == 7
    assert not hasattr(model, "class_weight")


def test_build_model_gradient_boosting_is_balanced_and_seeded() -> None:
    model = build_model("gradient_boosting", random_state=7)

    assert isinstance(model, HistGradientBoostingClassifier)
    assert model.random_state == 7
    assert model.class_weight == "balanced"
    assert model.early_stopping is False


def test_build_model_forwards_extra_kwargs() -> None:
    model = build_model("gradient_boosting", random_state=7, max_iter=50)

    assert model.max_iter == 50


def test_build_model_raises_for_unknown_model() -> None:
    with pytest.raises(NotImplementedError):
        build_model("unknown_model")


def test_train_model_fits_and_returns_the_model() -> None:
    model = build_model("gradient_boosting", random_state=7, max_iter=5)
    x_train = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y_train = ["benign", "attack", "benign", "attack"]

    fitted = train_model(model, x_train, y_train)

    assert fitted is model
    assert list(fitted.classes_) == ["attack", "benign"]
