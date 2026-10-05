"""Model catalog: factories for the four classifiers and their hyperparameter search spaces."""

from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier as _XGBClassifier

from xai_thesis.torch_mlp import TorchMLPClassifier


class _StringLabelXGBClassifier(_XGBClassifier):
    """XGBClassifier that accepts string class labels, like the other models in the catalog.

    XGBClassifier requires labels encoded as 0..n_classes-1. This subclass encodes them in fit()
    and decodes them in predict(), so classes_ exposes the original label strings. It is a
    subclass rather than a wrapper so that shap.TreeExplainer still recognizes the model.

    The encoder is attached only after super().fit() returns: xgboost validates the labels
    against classes_ during fit(), and at that point classes_ must still report the encoded
    integers.
    """

    def fit(self, X, y, **kwargs: Any) -> "_StringLabelXGBClassifier":
        encoder = LabelEncoder().fit(y)
        super().fit(X, encoder.transform(y), **kwargs)
        self._label_encoder = encoder
        return self

    @property
    def classes_(self) -> Any:
        encoder = self.__dict__.get("_label_encoder")
        return encoder.classes_ if encoder is not None else super().classes_

    def predict(self, X, **kwargs: Any) -> Any:
        return self._label_encoder.inverse_transform(super().predict(X, **kwargs).astype(int))


# All four models handle class imbalance by reweighting the training objective with the
# scikit-learn "balanced" weights n / (k * count_c); none resamples rows.
#   - random_forest, svm: class_weight="balanced".
#   - neural_network: TorchMLPClassifier applies the same per-class weights in its
#     cross-entropy loss (see torch_mlp.py).
#   - xgboost: has no class_weight parameter; pipeline.py passes the equivalent per-sample
#     weights to fit().
#
# svm uses a linear kernel, matching the configuration evaluated by Arreche et al. (2024); only
# C is searched. probability=True enables predict_proba, which the explanation methods require;
# it is overridable so that the hyperparameter search, which only calls predict(), can skip the
# costly probability calibration. max_iter is a safety cap that bounds the run time of a fit
# that fails to converge; it does not affect fits that converge normally.
#
# xgboost: tree_method="hist" for speed on the large training partitions; eval_metric is set to
# avoid a deprecation warning. Neither is searched.
_MODEL_FACTORIES: dict[str, Callable[..., ClassifierMixin]] = {
    "neural_network": lambda random_state, **kwargs: TorchMLPClassifier(random_state=random_state, **kwargs),
    "random_forest": lambda random_state, **kwargs: RandomForestClassifier(
        random_state=random_state, class_weight="balanced", **kwargs
    ),
    "svm": lambda random_state, **kwargs: SVC(
        random_state=random_state,
        class_weight="balanced",
        kernel="linear",
        **{"probability": True, "max_iter": 50_000, **kwargs},
    ),
    "xgboost": lambda random_state, **kwargs: _StringLabelXGBClassifier(
        random_state=random_state,
        **{"tree_method": "hist", "eval_metric": "mlogloss", **kwargs},
    ),
}

# Search spaces for RandomizedSearchCV, keyed by model name. The "model__" prefix addresses the
# ("model", ...) step of the preprocessing + model Pipeline built in pipeline.py.
#
# neural_network: learning rates are kept small because larger ones are unstable under the
# strong class weights. svm: only C is searched (gamma has no effect on a linear kernel); the
# grid stops at 10 because larger values combined with the class weights converge very slowly.
# xgboost: reg_lambda starts at 0.1 because unregularized boosting under extreme sample weights
# can degenerate.
PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "neural_network": {
        "model__hidden_layer_sizes": [(50,), (100,), (100, 50), (128, 64), (256, 128), (256, 128, 64)],
        "model__alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        "model__learning_rate_init": [0.00003, 0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.01],
        "model__batch_size": [128, 256, 512, 1024],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10, 20],
    },
    "svm": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 0.5, 1, 3, 10],
    },
    "xgboost": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 5, 7, 10, 15],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
    },
}


def build_model(model_name: str, random_state: int = 42, **kwargs: Any) -> ClassifierMixin:
    """Create a model by name."""
    try:
        factory = _MODEL_FACTORIES[model_name]
    except KeyError:
        raise NotImplementedError(f"Unknown or unimplemented model: {model_name}") from None
    return factory(random_state=random_state, **kwargs)


def train_model(model: ClassifierMixin, x_train, y_train, **fit_params: Any) -> ClassifierMixin:
    """Fit a model (or Pipeline) and return it. fit_params are forwarded to fit(), e.g.
    model__sample_weight for a Pipeline."""
    return model.fit(x_train, y_train, **fit_params)
