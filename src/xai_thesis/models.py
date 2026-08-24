from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier as _XGBClassifier


class _StringLabelXGBClassifier(_XGBClassifier):
    """XGBClassifier as of xgboost>=1.6 requires y already integer-encoded 0..n_classes-1 and
    raises ValueError otherwise (verified directly against the installed xgboost 3.4.1: fitting
    on this project's raw string labels, e.g. "Benign"/"Analysis", fails outright) - unlike
    every other model in this catalog, which is fit on those raw string labels straight from
    preprocessing.py's split_features_target, with .classes_ exposing them as-is afterward
    (evaluate_classifier, compute_shap_values, select_curated_instances, and _CLASS_OF_INTEREST's
    string lookups all assume this). This subclass (not a wrapper - shap.TreeExplainer needs to
    recognize the object directly, verified) label-encodes internally so it's a drop-in
    replacement matching that same string-label contract.

    Ordering matters in fit(): xgboost's own fit() validates np.unique(y) against self.classes_
    (a read-only property) *during* the call, so the encoder must not be visible on self yet at
    that point - it's assigned to self._label_encoder only after super().fit() returns
    successfully, and the classes_ property below falls back to the base (integer-arange)
    behavior via super().classes_ until then. Setting self._label_encoder up front instead
    would make classes_ report strings while xgboost's own internal check still expects the
    encoded integers, reproducing the exact ValueError this exists to avoid.
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


# MLPClassifier has no class_weight/sample_weight support, so it can't be rebalanced here
# like the rest; instead, pipeline.py resamples its training data (balance_classes) before
# fit. The other models get class_weight="balanced" given how imbalanced these datasets are.
#
# ⚠️ THESIS SCOPE, as of 2026-08-09: logistic_regression and gradient_boosting are out of
# scope for the thesis. Both stay fully implemented, trained, and explained here - nothing was
# removed - but the thesis now reports on only three models: neural_network, random_forest,
# and svm. Do not include logistic_regression/gradient_boosting results in methodology/results
# writing going forward unless the user explicitly reverses this (same pattern as DAPT2020/PDP
# being flagged out of scope elsewhere in this project - see README.md).
#
# ⚠️ xgboost added 2026-08-24, same out-of-scope-but-fully-functional treatment, at explicit
# user request/confirmation - not part of the thesis's own reported results. Unlike every other
# model here, XGBClassifier has no class_weight constructor param at all (not even an unsupported
# one to omit - it simply isn't in xgboost's sklearn API), only per-sample weights at fit() time.
# pipeline.py's _MODELS_REQUIRING_SAMPLE_WEIGHT computes those via
# sklearn.utils.class_weight.compute_sample_weight("balanced", y) - mathematically the same
# per-class weighting class_weight="balanced" applies internally for the other models - and
# passes them through at fit time, rather than reusing neural_network's resampling fallback
# below: resampling duplicates/drops rows and so shifts the effective class prior the model
# trains on (this project's own smote_balance_classes experiment, tried and reverted for
# neural_network, found evidence this prior shift - not the oversampling method - was the likely
# cause of a precision regression), whereas sample weighting reweights the loss without touching
# which rows exist. tree_method="hist" and eval_metric="mlogloss" are fixed for speed/determinism
# on this project's largest datasets (~2M rows) and to silence an eval_metric-at-fit-time
# deprecation warning respectively; neither is searched.
#
# random_forest was previously dropped from the catalog (shared gradient_boosting's
# tree-ensemble family and TreeExplainer, with near-highest cross-model SHAP agreement of
# any pair in this project, while underperforming it) - re-added by explicit request despite
# that redundancy concern still applying.
#
# svm here means SVC with a linear kernel, not the LinearSVC previously tried under this same
# catalog key and dropped for substantially worse macro-F1 than the other models. kernel stays
# fixed to match an external reference configuration exactly and is not searched; C is tuned
# via this project's usual RandomizedSearchCV as of 2026-08-09 (PARAM_DISTRIBUTIONS below) -
# the reference's own C=0.5 is kept only as tune_hyperparameters()'s starting model_params,
# overwritten once a search actually runs. An earlier RBF-kernel attempt was abandoned after
# measuring ~5.5 minutes for a single untuned fit on UNSW-NB15 alone (~101k rows) - the linear
# kernel was adopted for tractability as much as to match the reference.
#
# probability defaults to True (matching the reference configuration) but is deliberately left
# overridable via **kwargs (hence not hardcoded after it below, unlike every other fixed
# setting here) rather than baked in unconditionally: tune_hyperparameters() in pipeline.py
# passes probability=False during the search phase specifically, since RandomizedSearchCV's
# f1_macro scoring only ever calls .predict(), never .predict_proba() - Platt-scaling
# calibration is pure overhead there. probability=True is restored for the final fit in
# run_experiment() (every experiment config's model_params has no "probability" key, so the
# True default applies), which is the only place it's actually needed, for the explanation
# pipeline's predict_proba calls. sklearn 1.9 deprecates this parameter in favor of wrapping
# with CalibratedClassifierCV, but it still works and is simpler to keep matching the
# reference's stated configuration directly.
#
# max_iter=50_000 is a hard safety cap, not a modeling choice - discovered necessary the hard
# way: an early attempt at tuning C via RandomizedSearchCV (unsw_nb15, C candidates up to 100)
# hung for over 19 hours of CPU time on what should have been a multi-minute fit, almost
# certainly libsvm's QP solver failing to converge for some (C, fold) combination under
# class_weight="balanced"'s extreme per-sample weights on this project's severely imbalanced
# data - the same class of failure already documented for gradient_boosting's
# l2_regularization=0 (see PARAM_DISTRIBUTIONS below), just silent instead of producing an
# obviously-bad model. Without a cap, sklearn's default (-1, unlimited) has no way to fail fast
# on a pathological fit - it can only hang. With the cap, a fit that would have hung instead
# stops early with a ConvergenceWarning and returns whatever (possibly suboptimal) solution it
# had reached, which is a far better failure mode for an automated search loop that may run
# many fits unattended. 50,000 is generous headroom above every fit observed to actually
# converge normally (a handful of minutes each), while still bounding worst-case time to
# something finite.
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
    # C is the only thing searched: kernel="linear" and probability=True stay fixed to match
    # the external reference configuration (see _MODEL_FACTORIES above). gamma isn't included
    # since it has no effect on a linear kernel. Capped at 10, not higher: larger C forces a
    # harder-margin fit, which combined with class_weight="balanced"'s extreme per-sample
    # weights under this project's severe imbalance is the exact combination that triggered
    # the multi-hour libsvm non-convergence hang documented in _MODEL_FACTORIES above (that
    # incident used candidates up to 100) - max_iter=50_000 now bounds worst-case fit time
    # regardless, but there's no reason to spend search budget on C values this unlikely to be
    # both well-behaved and better than the reference's own C=0.5 anyway.
    "svm": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 0.5, 1, 3, 10],
    },
    # reg_lambda's minimum is 0.1, not 0: gradient_boosting's l2_regularization=0 incident above
    # showed zero regularization combined with extreme per-sample weights under this project's
    # severe imbalance can produce a near-degenerate model - xgboost's sample-weight-based
    # balancing (see _MODEL_FACTORIES' xgboost comment) creates the same kind of extreme weights,
    # so this grid avoids the analogous zero-regularization combination proactively rather than
    # rediscovering the same failure mode.
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
    """Fit a model (or Pipeline) and return it. fit_params are forwarded to .fit() as-is - e.g.
    sample_weight for a bare model, or model__sample_weight for a Pipeline (see pipeline.py's
    _MODELS_REQUIRING_SAMPLE_WEIGHT)."""
    return model.fit(x_train, y_train, **fit_params)
