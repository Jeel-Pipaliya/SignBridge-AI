"""
SignBridge AI - Configurable ISL Classifier Model Architecture (Week 4 Phase 3)
Implements Multi-Layer Perceptron (Dense + ReLU + Regularization + Softmax)
and ensemble architectures with dynamic input/output dimensions and hyperparameter tuning.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class ISLClassifier:
    """
    Configurable Machine Learning / Neural Network Classifier for Indian Sign Language.
    Automatically adapts output layer dimension to the number of classes.
    """

    def __init__(
        self,
        model_type: str = "mlp",
        input_dim: int = 63,
        num_classes: int = 32,
        hidden_units: Tuple[int, ...] = (128, 64),
        activation: str = "relu",
        alpha: float = 0.001,
        learning_rate_init: float = 0.001,
        learning_rate: str = "adaptive",
        max_iter: int = 200,
        random_state: int = 42,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
    ):
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_units = tuple(hidden_units)
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change

        self.model = self._build_model()
        self.classes_: Optional[np.ndarray] = None

    def _build_model(self) -> Any:
        """Construct the underlying estimator based on configuration."""
        if self.model_type == "mlp":
            # Multi-Layer Perceptron: Input -> Dense(128, ReLU) -> Dense(64, ReLU) -> Softmax(num_classes)
            return MLPClassifier(
                hidden_layer_sizes=self.hidden_units,
                activation=self.activation,
                solver="adam",
                alpha=self.alpha,
                batch_size=32,
                learning_rate=self.learning_rate,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                early_stopping=self.early_stopping,
                n_iter_no_change=self.n_iter_no_change,
                validation_fraction=0.15,
            )
        elif self.model_type in ("rf", "random_forest"):
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=14,
                min_samples_split=2,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "svm":
            return SVC(
                C=10.0,
                kernel="rbf",
                probability=True,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. Choose 'mlp', 'svm', or 'random_forest'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ISLClassifier":
        """Fit model to training dataset."""
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {X.shape[1]}")
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict discrete class index/label."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution via Softmax."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_d / np.sum(exp_d, axis=1, keepdims=True)
        else:
            preds = self.predict(X)
            probs = np.zeros((len(X), self.num_classes), dtype=np.float32)
            for i, p in enumerate(preds):
                probs[i, int(p)] = 1.0
            return probs

    def get_metadata(self) -> Dict[str, Any]:
        """Returns structural metadata and configuration dictionary."""
        return {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_units": list(self.hidden_units) if self.model_type == "mlp" else None,
            "activation": self.activation if self.model_type == "mlp" else None,
            "random_state": self.random_state,
            "n_features_in": getattr(self.model, "n_features_in_", self.input_dim),
            "n_outputs": getattr(self.model, "n_outputs_", self.num_classes),
            "loss_curve": list(getattr(self.model, "loss_curve_", [])),
        }


def build_isl_model(
    input_dim: int = 63,
    num_classes: int = 32,
    model_type: str = "mlp",
    hidden_units: Tuple[int, ...] = (128, 64),
    learning_rate: float = 0.001,
    random_seed: int = 42,
) -> ISLClassifier:
    """
    Factory API to build and return a configured ISLClassifier.
    """
    return ISLClassifier(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_units=hidden_units,
        learning_rate_init=learning_rate,
        random_state=random_seed,
    )
