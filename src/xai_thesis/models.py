from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# MLPClassifier has no class_weight/sample_weight support, so it trains unweighted;
# the rest get class_weight="balanced" given how imbalanced these datasets are.
_MODEL_FACTORIES: dict[str, Callable[..., ClassifierMixin]] = {
    "random_forest": lambda random_state, **kwargs: RandomForestClassifier(
        random_state=random_state, class_weight="balanced", **kwargs
    ),
    "svm": lambda random_state, **kwargs: LinearSVC(random_state=random_state, class_weight="balanced", **kwargs),
    "neural_network": lambda random_state, **kwargs: MLPClassifier(random_state=random_state, **kwargs),
    "gradient_boosting": lambda random_state, **kwargs: HistGradientBoostingClassifier(
        random_state=random_state, class_weight="balanced", **kwargs
    ),
}

# Hyperparameter search spaces for RandomizedSearchCV, keyed by model name. Parameter names
# are prefixed "model__" to match the ("model", ...) step name in the sklearn Pipeline that
# wraps preprocessing + model together (see pipeline.py).
PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 10, 20, 40],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", "log2"],
    },
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
        "model__l2_regularization": [0.0, 0.1, 1.0],
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
