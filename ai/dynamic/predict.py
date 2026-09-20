"""
SignBridge AI - Dynamic Gesture Predictor (Week 5)
Real-time sequence inference engine wrapping the trained Bi-LSTM model.
Processes (T=30, F=63) landmark sequences and outputs calibrated predictions.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.dynamic.model import BiLSTMClassifier
from ai.dynamic.preprocessing import compute_sequence_motion_energy
from ai.utils.logger import setup_logger

logger = setup_logger("DynamicPredictor")


class DynamicPredictionResult(NamedTuple):
    """Structured container for dynamic gesture inference output."""
    label: str
    confidence: float
    is_valid: bool
    motion_energy: float
    all_probabilities: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "is_valid": self.is_valid,
            "motion_energy": round(float(self.motion_energy), 4),
            "all_probabilities": {k: round(v, 4) for k, v in self.all_probabilities.items()},
        }


class DynamicSignPredictor:
    """
    Sub-millisecond inference for 30-frame dynamic landmark sequences.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = config.DYNAMIC_CONFIDENCE_THRESHOLD,
        min_motion_energy: float = 0.015,
    ):
        self.model_path = Path(model_path) if model_path is not None else config.DYNAMIC_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.min_motion_energy = min_motion_energy
        self.model: Optional[BiLSTMClassifier] = None
        self.classes: List[str] = config.DYNAMIC_CLASSES

        self._load_model()

    def _load_model(self) -> None:
        """Load serialized Bi-LSTM checkpoint if available."""
        if not self.model_path.exists():
            logger.warning(
                f"Dynamic model checkpoint not found at {self.model_path}. "
                "Dynamic inference will return invalid results until trained."
            )
            self.model = None
            return

        try:
            self.model = BiLSTMClassifier.load_checkpoint(self.model_path)
            self.classes = self.model.classes
            logger.info(f"DynamicSignPredictor loaded Bi-LSTM model with {len(self.classes)} classes: {self.classes}")
        except Exception as err:
            logger.error(f"Failed to load dynamic model checkpoint: {err}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict_sequence(self, sequence: np.ndarray) -> DynamicPredictionResult:
        """
        Run inference on sequence of shape (T, 63).
        """
        if self.model is None:
            return DynamicPredictionResult(
                label="...",
                confidence=0.0,
                is_valid=False,
                motion_energy=0.0,
                all_probabilities={},
            )

        seq = np.asarray(sequence, dtype=np.float32)
        motion_energy = compute_sequence_motion_energy(seq)

        # Gating: must exhibit dynamic motion to be considered a dynamic sign
        if motion_energy < self.min_motion_energy:
            return DynamicPredictionResult(
                label="...",
                confidence=0.0,
                is_valid=False,
                motion_energy=motion_energy,
                all_probabilities={},
            )

        probs = self.model.predict_proba(seq)[0]
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])
        pred_label = self.classes[max_idx]

        is_valid = confidence >= self.confidence_threshold
        prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}

        return DynamicPredictionResult(
            label=pred_label if is_valid else "...",
            confidence=confidence,
            is_valid=is_valid,
            motion_energy=motion_energy,
            all_probabilities=prob_dict,
        )
