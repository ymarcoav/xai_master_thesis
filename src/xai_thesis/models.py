from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# MLPClassifier has no class_weight/sample_weight support, so it can't be rebalanced here
# like the rest; instead, pipeline.py resamples its training data (balance_classes) before
# fit. The other models get class_weight="balanced" given how imbalanced these datasets are.
#
# random_forest was previously dropped from the catalog (shared gradient_boosting's
# tree-ensemble family and TreeExplainer, with near-highest cross-model SHAP agreement of
# any pair in this project, while underperforming it) - re-added by explicit request despite
# that redundancy concern still applying.
#
# svm here means SVC with a linear kernel, not the LinearSVC previously tried under this same
# catalog key and dropped for substantially worse macro-F1 than the other models. Configured
# to match an external reference exactly (kernel="linear", C=0.5 via model_params,
# probability=True) rather than tuned via this project's usual RandomizedSearchCV - no
# PARAM_DISTRIBUTIONS entry for it below, since these settings are meant to be adopted as-is,
# not searched around. An earlier RBF-kernel attempt was abandoned after measuring ~5.5
# minutes for a single untuned fit on UNSW-NB15 alone (~101k rows) - infeasible to tune, and
# likely infeasible to even fit once on CIC-IDS2017's ~2M rows. probability=True is kept
# despite the extra internal Platt-scaling cost it adds on top of the kernel fit, to match the
# reference configuration; sklearn 1.9 deprecates this in favor of wrapping with
# CalibratedClassifierCV, but the parameter still works and is simpler to keep matching the
# reference's stated configuration directly.
_MODEL_FACTORIES: dict[str, Callable[..., ClassifierMixin]] = {
    "logistic_regression": lambda random_state, **kwargs: LogisticRegression(
        random_state=random_state, class_weight="balanced", **kwargs
    ),
    "neural_network": lambda random_state, **kwargs: MLPClassifier(random_state=random_state, **kwargs),
    # early_stopping defaults to "auto" in sklearn, which does its own internal
    # stratified train/validation split; combined with hyperparameter-search subsampling
    # that can leave an already-rare class down to 1 member in a fold, which fails to
    # stratify. max_iter is already tuned directly, so early stopping is redundant here.
    "gradient_boosting": lambda random_state, **kwargs: HistGradientBoostingClassifier(
        random_state=random_state, class_weight="balanced", early_stopping=False, **kwargs
    ),
    "random_forest": lambda random_state, **kwargs: RandomForestClassifier(
        random_state=random_state, class_weight="balanced", **kwargs
    ),
    "svm": lambda random_state, **kwargs: SVC(
        random_state=random_state, class_weight="balanced", kernel="linear", probability=True, **kwargs
    ),
}

# Hyperparameter search spaces for RandomizedSearchCV, keyed by model name. Parameter names
# are prefixed "model__" to match the ("model", ...) step name in the sklearn Pipeline that
# wraps preprocessing + model together (see pipeline.py).
PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "model__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    },
    "neural_network": {
        "model__hidden_layer_sizes": [(50,), (100,), (100, 50), (128, 64), (256, 128), (256, 128, 64)],
        "model__alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        "model__learning_rate_init": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        "model__max_iter": [100, 200, 300, 500],
        "model__max_leaf_nodes": [7, 15, 31, 63, 100, 127],
        "model__max_depth": [None, 3, 5, 10, 15, 20],
        # l2_regularization=0.0 is excluded: HistGradientBoostingClassifier's leaf value is
        # sum(gradient)/(sum(hessian) + l2_regularization), and class_weight="balanced"
        # produces extreme per-sample weights under this project's severe class imbalance
        # (e.g. a ~334,000x weight ratio between CIC-IDS2017's Heartbleed and BENIGN
        # classes) - with zero regularization that combination produced a model with 3.7%
        # accuracy (vs. 99.5% with l2_regularization=0.1, identical everything else).
        "model__l2_regularization": [0.1, 0.3, 0.5, 0.7, 1.0],
        "model__min_samples_leaf": [10, 20, 30, 50],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10, 20],
    },
    # svm intentionally has no entry here: its hyperparameters are fixed to match an external
    # reference configuration directly (see _MODEL_FACTORIES above and each svm experiment
    # config's model_params), not searched via RandomizedSearchCV.
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
