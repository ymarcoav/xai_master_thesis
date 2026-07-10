from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# MLPClassifier has no class_weight/sample_weight support, so it trains unweighted;
# RF and the linear SVM get class_weight="balanced" given how imbalanced these datasets are.
_MODEL_FACTORIES: dict[str, Callable[..., ClassifierMixin]] = {
    "random_forest": lambda random_state, **kwargs: RandomForestClassifier(
        random_state=random_state, class_weight="balanced", **kwargs
    ),
    "svm": lambda random_state, **kwargs: LinearSVC(random_state=random_state, class_weight="balanced", **kwargs),
    "neural_network": lambda random_state, **kwargs: MLPClassifier(random_state=random_state, **kwargs),
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
