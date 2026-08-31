"""A small PyTorch MLP with a scikit-learn classifier interface.

Replaces ``sklearn.neural_network.MLPClassifier`` as this project's ``neural_network`` model
so that class imbalance can be handled the *same way* as every other model in the catalog -
by reweighting the training objective, never by resampling rows. ``MLPClassifier`` has no
``class_weight``/``sample_weight`` support, which previously forced a bespoke
resample-to-the-median-class-count path (removed alongside this module); ``nn.CrossEntropyLoss``
takes a per-class ``weight`` vector directly, using the exact ``n / (k * count_c)`` formula
``class_weight="balanced"`` applies for ``random_forest``/``svm`` and
``compute_sample_weight("balanced", ...)`` applies for ``xgboost``.

The wrapper keeps sklearn semantics so everything downstream is unchanged: ``fit(X, y)`` /
``predict(X)`` returning the original (string) labels / ``predict_proba(X)`` returning an
``(n_samples, n_classes)`` float64 array of softmax probabilities whose columns follow
``classes_`` order. It is a cloneable ``BaseEstimator`` (``RandomizedSearchCV`` relies on
this), and it consumes whatever the fitted ``ColumnTransformer`` hands it - it never fits or
touches the scaler/encoder itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _build_mlp(in_features: int, hidden_sizes: Sequence[int], n_classes: int) -> nn.Sequential:
    """Linear -> ReLU per hidden layer, then a final Linear to raw class logits (no softmax).

    ``hidden_sizes`` mirrors MLPClassifier's ``hidden_layer_sizes`` - one entry per hidden
    layer - so e.g. ``(256, 128)`` builds Linear(in,256) -> ReLU -> Linear(256,128) -> ReLU
    -> Linear(128, n_classes).
    """
    layers: list[nn.Module] = []
    previous = in_features
    for width in hidden_sizes:
        layers.append(nn.Linear(previous, width))
        layers.append(nn.ReLU())
        previous = width
    layers.append(nn.Linear(previous, n_classes))
    return nn.Sequential(*layers)


def _as_float32_array(x: Any) -> np.ndarray:
    """Densify (the ColumnTransformer can emit a sparse matrix) and cast to float32."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.ascontiguousarray(x, dtype=np.float32)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """PyTorch MLP classifier with class-weighted cross-entropy and a scikit-learn API.

    Parameter names match MLPClassifier where an equivalent exists so the existing
    ``PARAM_DISTRIBUTIONS["neural_network"]`` search space (``hidden_layer_sizes``, ``alpha``,
    ``learning_rate_init``) and experiment configs carry over unchanged:

    - ``hidden_layer_sizes``: one width per hidden layer.
    - ``alpha``: L2 penalty, passed to Adam as ``weight_decay``.
    - ``learning_rate_init``: Adam learning rate.
    - ``max_iter``: number of epochs.
    - ``class_weight``: ``"balanced"`` (default) applies ``n / (k * count_c)`` per-class weights
      to ``nn.CrossEntropyLoss``; ``None`` trains unweighted. No other value is accepted.
    - ``early_stopping`` / ``validation_fraction`` / ``n_iter_no_change`` / ``tol``: hold out a
      stratified validation split and stop when validation loss fails to improve by more than
      ``tol`` for ``n_iter_no_change`` consecutive epochs, restoring the best weights.
    - ``batch_size``: DataLoader minibatch size (not part of the search space).
    - ``random_state``: seeds torch, the parameter init, the train/val split and the DataLoader
      shuffle generator, and enables ``torch.use_deterministic_algorithms`` so explanation
      results are reproducible run to run.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (100,),
        *,
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        class_weight: str | None = "balanced",
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
        batch_size: int = 256,
        random_state: int | None = None,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state

    # ------------------------------------------------------------------ fit / predict

    def fit(self, X: Any, y: Any) -> "TorchMLPClassifier":
        seed = 0 if self.random_state is None else int(self.random_state)
        torch.manual_seed(seed)
        # warn_only: some ops have no deterministic CPU kernel; we want reproducibility where
        # possible without a hard failure on those.
        torch.use_deterministic_algorithms(True, warn_only=True)

        x_array = _as_float32_array(X)
        self._label_encoder = LabelEncoder().fit(np.asarray(y))
        self.classes_ = self._label_encoder.classes_
        y_encoded = self._label_encoder.transform(np.asarray(y)).astype(np.int64)
        n_classes = len(self.classes_)
        self.n_features_in_ = x_array.shape[1]

        # class_weight="balanced": w_c = n_samples / (n_classes * count_c), computed on THIS y
        # (the training split the Pipeline passes in - no separate held-out counts). bincount is
        # in encoded-label order, so weight[c] lines up with CrossEntropyLoss's target integer c
        # and with self.classes_[c]. Getting this indexing wrong is the classic bug in this kind
        # of change; the uniform-vs-balanced recall check in the tests guards it.
        class_counts = np.bincount(y_encoded, minlength=n_classes)
        if self.class_weight == "balanced":
            weights = len(y_encoded) / (n_classes * np.clip(class_counts, 1, None))
            weight_tensor: torch.Tensor | None = torch.as_tensor(weights, dtype=torch.float32)
        elif self.class_weight is None:
            weight_tensor = None
        else:
            raise ValueError(f"class_weight must be 'balanced' or None, got {self.class_weight!r}")

        if self.early_stopping and n_classes < len(y_encoded):
            stratify = y_encoded if np.min(class_counts) >= 2 else None
            train_idx, val_idx = train_test_split(
                np.arange(len(y_encoded)),
                test_size=self.validation_fraction,
                random_state=seed,
                stratify=stratify,
            )
        else:
            train_idx, val_idx = np.arange(len(y_encoded)), np.empty(0, dtype=int)

        x_train_t = torch.from_numpy(x_array[train_idx])
        y_train_t = torch.from_numpy(y_encoded[train_idx])

        model = _build_mlp(self.n_features_in_, tuple(self.hidden_layer_sizes), n_classes)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha
        )

        generator = torch.Generator()
        generator.manual_seed(seed)
        batch_size = min(self.batch_size, len(train_idx)) or 1
        loader = DataLoader(
            TensorDataset(x_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        x_val_t = torch.from_numpy(x_array[val_idx]) if len(val_idx) else None
        y_val_t = torch.from_numpy(y_encoded[val_idx]) if len(val_idx) else None

        best_val = np.inf
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        self.n_iter_ = 0

        for _ in range(self.max_iter):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            self.n_iter_ += 1

            if x_val_t is None:
                continue
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(x_val_t), y_val_t))
            if val_loss < best_val - self.tol:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.n_iter_no_change:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_fitted()
        with torch.no_grad():
            logits = self._model(torch.from_numpy(_as_float32_array(X)))
            proba = torch.softmax(logits, dim=1)
        return proba.numpy().astype(np.float64)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # ------------------------------------------------------------------ internals

    def _check_fitted(self) -> None:
        if not hasattr(self, "_model"):
            raise RuntimeError("TorchMLPClassifier instance is not fitted yet; call fit() first.")
