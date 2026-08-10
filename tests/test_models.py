import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from xai_thesis.models import build_model, train_model


def test_build_model_logistic_regression_is_balanced_and_seeded() -> None:
    model = build_model("logistic_regression", random_state=7)

    assert isinstance(model, LogisticRegression)
    assert model.random_state == 7
    assert model.class_weight == "balanced"


def test_build_model_logistic_regression_forwards_kwargs() -> None:
    model = build_model("logistic_regression", random_state=7, C=10, max_iter=5000)

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


def test_build_model_random_forest_is_balanced_and_seeded() -> None:
    model = build_model("random_forest", random_state=7)

    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == 7
    assert model.class_weight == "balanced"


def test_build_model_svm_is_linear_kernel_balanced_seeded_with_probability() -> None:
    model = build_model("svm", random_state=7)

    assert isinstance(model, SVC)
    assert model.random_state == 7
    assert model.class_weight == "balanced"
    assert model.kernel == "linear"
    assert model.probability is True
    # Hard safety cap, not a modeling choice - without it, a pathological (C, fold)
    # combination hung for 19+ hours during a real tuning run instead of failing fast (see
    # models.py). Bounds worst-case fit time; well-behaved fits never get close to it.
    assert model.max_iter == 50_000


def test_build_model_svm_probability_and_max_iter_are_overridable() -> None:
    # tune_hyperparameters() overrides probability=False during search (Platt-scaling is
    # unneeded overhead there - see pipeline.py); confirms **kwargs can override these two
    # fixed defaults specifically, unlike kernel/class_weight, which never vary.
    model = build_model("svm", random_state=7, probability=False, max_iter=1000)

    assert model.probability is False
    assert model.max_iter == 1000


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
