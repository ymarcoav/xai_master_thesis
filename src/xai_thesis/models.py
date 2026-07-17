from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# MLPClassifier has no class_weight/sample_weight support, so it can't be rebalanced here
# like the rest; instead, pipeline.py resamples its training data (balance_classes) before
# fit. The other two get class_weight="balanced" given how imbalanced these datasets are.
#
# svm (LinearSVC) has no predict_proba, which some explainability techniques want. We
# deliberately don't wrap it in CalibratedClassifierCV here: calibrating a multiclass
# one-vs-rest decision function per class and renormalizing doesn't preserve the original
# argmax ranking under severe class imbalance, which measurably degraded macro F1 in
# testing (e.g. 0.43->0.33 on UNSW-NB15, same C value). Any probability estimate needed
# for explainability is produced locally there, on a calibrated copy, never as the model
# of record here.
#
# random_forest was dropped from the catalog: it shares gradient_boosting's tree-ensemble
# family and TreeExplainer (near-highest cross-model SHAP agreement of any pair in this
# project), while consistently underperforming it on 2 of 3 datasets and only tying on the
# third - little unique signal lost, and it simplifies the model x technique combinatorics
# for the expert-interview stimulus set.
_MODEL_FACTORIES: dict[str, Callable[..., ClassifierMixin]] = {
    "svm": lambda random_state, **kwargs: LinearSVC(random_state=random_state, class_weight="balanced", **kwargs),
    "neural_network": lambda random_state, **kwargs: MLPClassifier(random_state=random_state, **kwargs),
    # early_stopping defaults to "auto" in sklearn, which does its own internal
    # stratified train/validation split; combined with hyperparameter-search subsampling
    # that can leave an already-rare class down to 1 member in a fold, which fails to
    # stratify. max_iter is already tuned directly, so early stopping is redundant here.
    "gradient_boosting": lambda random_state, **kwargs: HistGradientBoostingClassifier(
        random_state=random_state, class_weight="balanced", early_stopping=False, **kwargs
    ),
}

# Hyperparameter search spaces for RandomizedSearchCV, keyed by model name. Parameter names
# are prefixed "model__" to match the ("model", ...) step name in the sklearn Pipeline that
# wraps preprocessing + model together (see pipeline.py).
PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "svm": {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    },
    "neural_network": {
        "model__hidden_layer_sizes": [(50,), (100,), (100, 50), (128, 64)],
        "model__alpha": [0.0001, 0.001, 0.01, 0.1],
        "model__learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 300],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__max_depth": [None, 5, 10, 20],
        # l2_regularization=0.0 is excluded: HistGradientBoostingClassifier's leaf value is
        # sum(gradient)/(sum(hessian) + l2_regularization), and class_weight="balanced"
        # produces extreme per-sample weights under this project's severe class imbalance
        # (e.g. a ~334,000x weight ratio between CIC-IDS2017's Heartbleed and BENIGN
        # classes) - with zero regularization that combination produced a model with 3.7%
        # accuracy (vs. 99.5% with l2_regularization=0.1, identical everything else).
        "model__l2_regularization": [0.1, 0.5, 1.0],
    },
}


def build_model(model_name: str, random_state: int = 42, **kwargs: Any) -> ClassifierMixin:
    """Create a model by name."""
    try:
        factory = _MODEL_FACTORIES[model_name]
    except KeyError:
        raise NotImplementedError(f"Unknown or unimplemented model: {model_name}") from None
    return factory(random_state=random_state, **kwargs)


def train_model(model: ClassifierMixin, x_train, y_train) -> ClassifierMixin:
    """Fit a model and return it."""
    return model.fit(x_train, y_train)
