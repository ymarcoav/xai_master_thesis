import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xai_thesis.models import _StringLabelXGBClassifier, build_model, train_model
from xai_thesis.torch_mlp import TorchMLPClassifier


def test_build_model_neural_network_is_torch_mlp_with_balanced_class_weight() -> None:
    model = build_model("neural_network", random_state=7)

    assert isinstance(model, TorchMLPClassifier)
    assert model.random_state == 7
    # imbalance is handled inside CrossEntropyLoss the same n/(k*count) way as the other
    # models' class_weight="balanced" - not by resampling.
    assert model.class_weight == "balanced"


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
    model = build_model("random_forest", random_state=7, n_estimators=50)

    assert model.n_estimators == 50


def test_build_model_xgboost_is_seeded_with_hist_tree_method() -> None:
    model = build_model("xgboost", random_state=7)

    assert isinstance(model, _StringLabelXGBClassifier)
    assert model.random_state == 7
    assert model.tree_method == "hist"
    # No class_weight: xgboost's XGBClassifier has no such constructor param at all, unlike
    # every class_weight="balanced" model above - pipeline.py handles imbalance via
    # sample_weight at fit time instead (_MODELS_REQUIRING_SAMPLE_WEIGHT).
    assert not hasattr(model, "class_weight")


def test_build_model_xgboost_forwards_kwargs() -> None:
    model = build_model("xgboost", random_state=7, n_estimators=10, max_depth=3)

    assert model.n_estimators == 10
    assert model.max_depth == 3


def test_xgboost_fits_and_predicts_original_string_labels() -> None:
    # XGBClassifier itself rejects string labels; the wrapper must accept them, and classes_,
    # predict() and predict_proba's column order must all agree with the original labels.
    model = build_model("xgboost", random_state=7, n_estimators=5, max_depth=2)
    x_train = [[0, 0], [1, 1], [0, 1], [1, 0], [2, 2], [2, 3]]
    y_train = ["benign", "attack", "benign", "attack", "other", "other"]

    fitted = train_model(model, x_train, y_train)

    assert fitted is model
    assert list(fitted.classes_) == ["attack", "benign", "other"]
    predictions = fitted.predict(x_train)
    assert set(predictions) <= {"attack", "benign", "other"}
    proba = fitted.predict_proba(x_train)
    assert proba.shape == (6, 3)


def test_build_model_raises_for_unknown_model() -> None:
    with pytest.raises(NotImplementedError):
        build_model("unknown_model")


def test_train_model_fits_and_returns_the_model() -> None:
    model = build_model("random_forest", random_state=7, n_estimators=5)
    x_train = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y_train = ["benign", "attack", "benign", "attack"]

    fitted = train_model(model, x_train, y_train)

    assert fitted is model
    assert list(fitted.classes_) == ["attack", "benign"]
