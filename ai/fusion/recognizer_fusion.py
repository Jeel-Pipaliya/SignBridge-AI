"""
SignBridge AI - Multi-Modal Static + Dynamic Recognizer Fusion Engine (Week 5)
Arbitrates between single-frame static classifier and temporal Bi-LSTM gesture model.

Design:
  1. Hand Landmark Extraction & Normalization
  2. Static prediction (evaluated every frame, 5-frame temporal smoothing)
  3. Dynamic sequence buffering (evaluated every K frames with motion energy gating)
  4. Decision Layer: prioritize high-confidence dynamic gestures with verified movement,
     default to temporally stabilized static signs.
  5. Cooldown & Debounce management to prevent duplicate commits.
"""

import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple, Union
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.predict import SignPredictor, PredictionResult
from ai.dynamic.predict import DynamicSignPredictor, DynamicPredictionResult
from ai.preprocessing.normalization import normalize_landmarks
from ai.preprocessing.sequence_builder import SequenceBuilder
from realtime.recognizer import TemporalSmoother
from ai.utils.logger import setup_logger

logger = setup_logger("RecognizerFusion")


class RecognitionResult(NamedTuple):
    """Unified recognition container across static and dynamic modalities."""
    type: str  # "STATIC", "DYNAMIC", or "NONE"
    label: str
    confidence: float
    timestamp: float
    is_stable: bool
    motion_energy: float

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "timestamp": round(float(self.timestamp), 4),
            "is_stable": self.is_stable,
            "motion_energy": round(float(self.motion_energy), 4),
        }


class StaticDynamicFusionEngine:
    """
    Coordinates static frame predictions and temporal sequence gesture recognition.
    """

    def __init__(
        self,
        static_predictor: Optional[SignPredictor] = None,
        dynamic_predictor: Optional[DynamicSignPredictor] = None,
        sequence_length: int = config.DYNAMIC_SEQUENCE_LENGTH,
        inference_interval: int = config.DYNAMIC_INFERENCE_INTERVAL,
        dynamic_cooldown_seconds: float = config.DYNAMIC_COOLDOWN_SECONDS,
        static_threshold: float = config.CONFIDENCE_THRESHOLD,
        dynamic_threshold: float = config.DYNAMIC_CONFIDENCE_THRESHOLD,
        min_motion_energy: float = 0.02,
    ):
        self.sequence_length = sequence_length
        self.inference_interval = inference_interval
        self.dynamic_cooldown_seconds = dynamic_cooldown_seconds
        self.static_threshold = static_threshold
        self.dynamic_threshold = dynamic_threshold
        self.min_motion_energy = min_motion_energy

        # Initialize or reuse predictors
        self.static_predictor = static_predictor or SignPredictor(confidence_threshold=static_threshold)
        self.dynamic_predictor = dynamic_predictor or DynamicSignPredictor(
            confidence_threshold=dynamic_threshold,
            min_motion_energy=min_motion_energy,
        )

        # Buffers
        self.sequence_builder = SequenceBuilder(sequence_length=sequence_length)
        self.static_smoother = TemporalSmoother(window_size=config.SMOOTHING_WINDOW)

        # State tracking
        self.frame_count: int = 0
        self._last_dynamic_commit_time: float = 0.0
        self._last_dynamic_label: Optional[str] = None

    def process_landmarks(
        self,
        landmarks: Optional[Union[List, np.ndarray]],
    ) -> RecognitionResult:
        """
        Process single-frame hand landmarks through the full fusion hierarchy.

        Args:
            landmarks: Raw MediaPipe landmarks or pre-normalized 63D vector.

        Returns:
            RecognitionResult with modality type, label, confidence, and metrics.
        """
        now = time.time()
        self.frame_count += 1

        if landmarks is None or len(landmarks) == 0:
            # Hand missing: flush or decay buffers gracefully
            self.static_smoother.update(PredictionResult("...", "...", 0.0, False, {}))
            return RecognitionResult(
                type="NONE",
                label="...",
                confidence=0.0,
                timestamp=now,
                is_stable=False,
                motion_energy=0.0,
            )

        # 1. Normalize 63D landmark features
        norm_features = normalize_landmarks(landmarks)
        self.sequence_builder.add_frame(norm_features)

        # 2. Static Prediction & 5-Frame Temporal Smoothing
        raw_static = self.static_predictor.predict_features(norm_features)
        smoothed_static_sign, static_conf = self.static_smoother.update(raw_static)

        # 3. Dynamic Gesture Prediction Evaluation
        dynamic_triggered = False
        dynamic_result = None

        if self.sequence_builder.is_ready() and (self.frame_count % self.inference_interval == 0):
            seq = self.sequence_builder.get_sequence()
            if seq is not None and self.dynamic_predictor.is_ready:
                pred = self.dynamic_predictor.predict_sequence(seq)
                if pred.is_valid and pred.label in config.DYNAMIC_CLASSES:
                    # Check dynamic debounce cooldown
                    time_since_dynamic = now - self._last_dynamic_commit_time
                    if (pred.label != self._last_dynamic_label) or (time_since_dynamic >= self.dynamic_cooldown_seconds):
                        dynamic_triggered = True
                        dynamic_result = pred
                        self._last_dynamic_commit_time = now
                        self._last_dynamic_label = pred.label
                        logger.info(f"Dynamic sign detected: {pred.label} (Conf: {pred.confidence:.1%}, Energy: {pred.motion_energy:.3f})")

        # 4. Decision Arbitration Layer
        if dynamic_triggered and dynamic_result is not None:
            # Dynamic sign takes precedence over static frames during active gestures
            return RecognitionResult(
                type="DYNAMIC",
                label=dynamic_result.label,
                confidence=dynamic_result.confidence,
                timestamp=now,
                is_stable=True,
                motion_energy=dynamic_result.motion_energy,
            )

        # Default to stabilized static sign
        is_static_stable = (smoothed_static_sign != "...") and (static_conf >= self.static_threshold)
        return RecognitionResult(
            type="STATIC" if is_static_stable else "NONE",
            label=smoothed_static_sign if is_static_stable else "...",
            confidence=static_conf,
            timestamp=now,
            is_stable=is_static_stable,
            motion_energy=0.0,
        )

    def reset(self) -> None:
        """Clear temporal buffers."""
        self.sequence_builder.reset()
        self.static_smoother.reset()
        self._last_dynamic_label = None
