"""
SignBridge AI - Temporal Sequence Models (Week 5)
Implements:
  1. BiLSTMClassifier: Bidirectional LSTM neural network with Dropout, Dense ReLU,
     and Softmax classification for dynamic signs (HELLO, J, Z).
  2. DynamicBaselineClassifier: Lightweight baseline using temporal statistics
     (mean, std, min, max, velocity) + Random Forest / MLP classifier.
"""

import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.utils.logger import setup_logger

logger = setup_logger("DynamicModel")

DEFAULT_CLASSES = ["HELLO", "J", "Z"]


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM architecture for dynamic gesture classification.
    Topology:
      Input: (Batch, Sequence Length = 30, Features = 63)
      -> Bi-LSTM Layer 1: 128 units (bidirectional -> 256)
      -> Dropout(0.3)
      -> Bi-LSTM Layer 2: 64 units (bidirectional -> 128)
      -> Dense: 64 units, ReLU
      -> Dropout(0.3)
      -> Linear: num_classes (Softmax)
    """

    def __init__(
        self,
        sequence_length: int = 30,
        feature_dim: int = 63,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        classes: Optional[List[str]] = None,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.classes = classes if classes is not None else DEFAULT_CLASSES

        # Layer 1: Bidirectional LSTM
        self.lstm1 = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: Bidirectional LSTM
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout)

        # Dense projection and classification head
        self.fc1 = nn.Linear(64 * 2, 64)
        self.relu = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, T, F) or (T, F).
        Returns:
            Logits of shape (Batch, num_classes).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Shape (1, T, F)

        # LSTM 1
        out1, _ = self.lstm1(x)
        out1 = self.drop1(out1)

        # LSTM 2
        out2, _ = self.lstm2(out1)
        out2 = self.drop2(out2)

        # Last temporal step representation
        last_step = out2[:, -1, :]  # Shape (Batch, 128)

        # Dense head
        dense = self.relu(self.fc1(last_step))
        dense = self.drop3(dense)
        logits = self.out(dense)
        return logits

    def predict_proba(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict probability distribution over dynamic classes.
        Returns:
            np.ndarray of shape (Batch, num_classes) with normalized probabilities.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                t_x = torch.from_numpy(x).float()
            else:
                t_x = x.float()

            if t_x.dim() == 2:
                t_x = t_x.unsqueeze(0)

            logits = self.forward(t_x)
            probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class indices.
        """
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=-1)

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_type": "BiLSTM",
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout_rate,
            "classes": self.classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        val_acc: float = 0.0,
    ) -> None:
        """Save state dictionary and training metadata."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.state_dict(),
            "config": self.get_metadata(),
            "epoch": epoch,
            "val_acc": val_acc,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved Bi-LSTM checkpoint to {filepath} (Val Acc: {val_acc:.2%})")

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path], device: str = "cpu") -> "BiLSTMClassifier":
        """Load trained model from checkpoint file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dynamic model checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)
        meta = checkpoint.get("config", {})
        model = cls(
            sequence_length=meta.get("sequence_length", 30),
            feature_dim=meta.get("feature_dim", 63),
            hidden_size=meta.get("hidden_size", 128),
            num_layers=meta.get("num_layers", 2),
            num_classes=meta.get("num_classes", 3),
            dropout=meta.get("dropout", 0.3),
            classes=meta.get("classes", DEFAULT_CLASSES),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        return model


class DynamicBaselineClassifier:
    """
    Lightweight baseline classifier for dynamic gesture comparison.
    Extracts temporal summary statistics over each sequence:
      [Mean (63), Std (63), Min (63), Max (63), Delta/Velocity (63)] -> Total 315 features
    Trained with a Random Forest or MLP classifier.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        classes: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.classes = classes if classes is not None else DEFAULT_CLASSES
        self.random_state = random_state

        if model_type in ("random_forest", "rf"):
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=random_state,
            )
        elif model_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=150,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown baseline model_type: {model_type}")

    @staticmethod
    def extract_temporal_features(sequence: np.ndarray) -> np.ndarray:
        """
        Extract summary statistics vector from a (T, F) sequence array.
        Returns: 1D array of shape (5 * F = 315,).
        """
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim == 3:
            # Batch of sequences (N, T, F)
            feats = []
            for s in seq:
                feats.append(DynamicBaselineClassifier.extract_temporal_features(s))
            return np.array(feats, dtype=np.float32)

        mean_f = np.mean(seq, axis=0)
        std_f = np.std(seq, axis=0)
        min_f = np.min(seq, axis=0)
        max_f = np.max(seq, axis=0)
        vel_f = np.mean(np.diff(seq, axis=0), axis=0) if len(seq) > 1 else np.zeros_like(mean_f)

        return np.concatenate([mean_f, std_f, min_f, max_f, vel_f]).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DynamicBaselineClassifier":
        """
        Fit baseline on sequences X: shape (N, T, F) or extracted features (N, 315).
        """
        if X.ndim == 3:
            feats = self.extract_temporal_features(X)
        else:
            feats = X
        self.model.fit(feats, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2 and X.shape[1] == 63:
            # Single sequence (T=30, F=63)
            feats = self.extract_temporal_features(X).reshape(1, -1)
        elif X.ndim == 3:
            feats = self.extract_temporal_features(X)
        else:
            feats = X
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)
        return self.model.predict_proba(feats)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_type": f"Baseline_{self.model_type}",
            "classes": self.classes,
            "feature_dim": 315,
        }
