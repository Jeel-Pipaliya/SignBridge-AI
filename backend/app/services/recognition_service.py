"""
SignBridge AI - Recognition Service
Coordinates real-time static 32-class ISL inference and dynamic Bi-LSTM sequence inference.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.app.core.config import settings
from backend.app.schemas.recognition import (
    CandidatePrediction,
    LandmarkPoint,
    PredictionResponse,
)
from ai.feature_extraction import normalize_landmarks
from ai.predict import SignPredictor
from ai.dynamic.predict import DynamicSignPredictor

logger = logging.getLogger("RecognitionService")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class RecognitionService:
    """Service wrapping real static and dynamic ISL ML models."""

    def __init__(self):
        self._static_predictor: Optional[SignPredictor] = None
        self._dynamic_predictor: Optional[DynamicSignPredictor] = None
        self._static_metadata: Dict[str, Any] = {}
        self._dynamic_metadata: Dict[str, Any] = {}
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Load static and dynamic models from configured paths."""
        # 1. Static ISL Predictor
        static_model_path = PROJECT_ROOT / settings.STATIC_MODEL_PATH
        if not static_model_path.exists():
            # Fallback to models/isl_classifier.pkl if best doesn't exist
            static_model_path = PROJECT_ROOT / "models" / "isl_classifier.pkl"

        encoder_path = PROJECT_ROOT / "models" / "label_encoder.pkl"
        try:
            if static_model_path.exists() and encoder_path.exists():
                self._static_predictor = SignPredictor(
                    model_path=static_model_path,
                    encoder_path=encoder_path,
                    confidence_threshold=settings.CONFIDENCE_THRESHOLD,
                )
                logger.info(f"Loaded Static ISL Classifier: {len(self._static_predictor.classes)} classes")
            else:
                logger.warning(f"Static model file or encoder not found at {static_model_path}")
        except Exception as err:
            logger.error(f"Failed to initialize static model: {err}")

        # 2. Dynamic ISL Bi-LSTM Predictor
        dynamic_model_path = PROJECT_ROOT / settings.DYNAMIC_MODEL_PATH
        try:
            if dynamic_model_path.exists():
                self._dynamic_predictor = DynamicSignPredictor(
                    model_path=dynamic_model_path,
                    confidence_threshold=0.75,
                )
                logger.info("Loaded Dynamic ISL Bi-LSTM Predictor")
            else:
                logger.warning(f"Dynamic model file not found at {dynamic_model_path}")
        except Exception as err:
            logger.error(f"Failed to initialize dynamic model: {err}")

        # 3. Load Metadata JSONs
        meta_path = PROJECT_ROOT / settings.STATIC_METADATA_PATH
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._static_metadata = json.load(f)
            except Exception as err:
                logger.warning(f"Error reading static metadata: {err}")

        dyn_meta_path = PROJECT_ROOT / settings.DYNAMIC_METADATA_PATH
        if dyn_meta_path.exists():
            try:
                with open(dyn_meta_path, "r", encoding="utf-8") as f:
                    self._dynamic_metadata = json.load(f)
            except Exception as err:
                logger.warning(f"Error reading dynamic metadata: {err}")

    @property
    def is_static_available(self) -> bool:
        return self._static_predictor is not None

    @property
    def is_dynamic_available(self) -> bool:
        return self._dynamic_predictor is not None

    @property
    def supported_classes(self) -> List[str]:
        if self._static_predictor:
            return self._static_predictor.classes
        return self._static_metadata.get("classes", [])

    @property
    def dynamic_classes(self) -> List[str]:
        if self._dynamic_predictor:
            return self._dynamic_predictor.classes
        return self._dynamic_metadata.get("classes", ["HELLO", "J", "Z"])

    def get_metadata(self) -> Dict[str, Any]:
        """Returns consolidated metadata for both static and dynamic models."""
        return {
            "static_model": {
                "available": self.is_static_available,
                "model_name": self._static_metadata.get("model_name", "SignBridge_ISL_MLP_NeuralNet"),
                "version": self._static_metadata.get("version", "1.0"),
                "architecture": self._static_metadata.get("model_type", "mlp").upper(),
                "num_classes": len(self.supported_classes),
                "classes": self.supported_classes,
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
                "train_accuracy": self._static_metadata.get("train_accuracy", 0.7308),
                "val_accuracy": self._static_metadata.get("validation_accuracy", 0.6905),
                "test_accuracy": self._static_metadata.get("test_accuracy", 0.6607),
                "test_macro_f1": self._static_metadata.get("test_macro_f1", 0.6085),
            },
            "dynamic_model": {
                "available": self.is_dynamic_available,
                "model_name": "SignBridge_Dynamic_BiLSTM",
                "architecture": "BiLSTM",
                "classes": self.dynamic_classes,
                "sequence_length": 30,
                "test_accuracy": self._dynamic_metadata.get("test_accuracy", 1.0),
                "test_macro_f1": self._dynamic_metadata.get("test_macro_f1", 1.0),
            },
        }

    def _convert_to_feature_vector(
        self,
        raw_landmarks: Union[List[LandmarkPoint], List[List[float]], List[float]]
    ) -> np.ndarray:
        """Converts raw landmarks (list of dicts/points or list of coords) to invariant 63D vector."""
        if len(raw_landmarks) == 21:
            # 21 3D points
            pts = []
            for item in raw_landmarks:
                if isinstance(item, LandmarkPoint):
                    pts.append((item.x, item.y, item.z))
                elif isinstance(item, dict):
                    pts.append((float(item.get("x", 0.0)), float(item.get("y", 0.0)), float(item.get("z", 0.0))))
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    pts.append((float(item[0]), float(item[1]), float(item[2])))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    pts.append((float(item[0]), float(item[1]), 0.0))
                else:
                    raise ValueError(f"Unrecognized landmark point format: {item}")
            
            # Apply invariant translation and scale normalization
            norm_vec = normalize_landmarks(pts)
            return norm_vec
        elif len(raw_landmarks) == 63:
            # Flat 63-element list
            return np.array(raw_landmarks, dtype=np.float32)
        else:
            raise ValueError(
                f"Expected 21 3D points or 63 flat coordinates, got {len(raw_landmarks)} items"
            )

    def predict_landmarks(
        self,
        landmarks: Union[List[LandmarkPoint], List[List[float]], List[float]],
        handedness: Optional[str] = "Right",
        confidence_threshold: Optional[float] = None,
    ) -> PredictionResponse:
        """Executes real-time static sign classification on landmark payload."""
        threshold = confidence_threshold if confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD
        start_time = time.perf_counter()

        if not self.is_static_available:
            return PredictionResponse(
                predicted_sign="Unknown",
                confidence=0.0,
                is_recognized=False,
                is_supported_vocabulary=False,
                model_name="SignBridge_ISL_MLP_NeuralNet",
                model_version="1.0",
                top_candidates=[],
                inference_duration_ms=0.0,
                handedness=handedness,
            )

        try:
            features = self._convert_to_feature_vector(landmarks)
            result = self._static_predictor.predict_features(features)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Rank top candidates
            candidates: List[CandidatePrediction] = []
            if result.all_probabilities:
                sorted_probs = sorted(
                    result.all_probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
                for sign, conf in sorted_probs[:5]:
                    candidates.append(CandidatePrediction(sign=sign, confidence=conf))

            is_valid = result.confidence >= threshold
            predicted_label = result.label if is_valid else "Unknown"

            return PredictionResponse(
                predicted_sign=predicted_label,
                confidence=float(result.confidence),
                is_recognized=is_valid,
                is_supported_vocabulary=result.raw_label in self.supported_classes,
                model_name=self._static_metadata.get("model_name", "SignBridge_ISL_MLP_NeuralNet"),
                model_version=self._static_metadata.get("version", "1.0"),
                top_candidates=candidates,
                inference_duration_ms=round(duration_ms, 2),
                handedness=handedness,
            )
        except Exception as err:
            logger.error(f"Inference error in predict_landmarks: {err}")
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return PredictionResponse(
                predicted_sign="Unknown",
                confidence=0.0,
                is_recognized=False,
                is_supported_vocabulary=False,
                model_name="SignBridge_ISL_MLP_NeuralNet",
                model_version="1.0",
                top_candidates=[],
                inference_duration_ms=round(duration_ms, 2),
                handedness=handedness,
            )

    def predict_sequence(
        self,
        sequence: List[Union[List[LandmarkPoint], List[List[float]], List[float]]],
        confidence_threshold: Optional[float] = None,
    ) -> PredictionResponse:
        """Executes real-time temporal Bi-LSTM gesture recognition on sequence buffer."""
        threshold = confidence_threshold if confidence_threshold is not None else 0.75
        start_time = time.perf_counter()

        if not self.is_dynamic_available or len(sequence) == 0:
            return PredictionResponse(
                predicted_sign="Unknown",
                confidence=0.0,
                is_recognized=False,
                is_supported_vocabulary=False,
                model_name="SignBridge_Dynamic_BiLSTM",
                model_version="1.0",
                top_candidates=[],
                inference_duration_ms=0.0,
            )

        try:
            # Process frames into (T, 63)
            frames = []
            for frame_landmarks in sequence:
                feat = self._convert_to_feature_vector(frame_landmarks)
                frames.append(feat)

            seq_array = np.array(frames, dtype=np.float32)  # (T, 63)
            dyn_result = self._dynamic_predictor.predict_sequence(seq_array)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            candidates: List[CandidatePrediction] = []
            if dyn_result.all_probabilities:
                sorted_probs = sorted(
                    dyn_result.all_probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
                for sign, conf in sorted_probs:
                    candidates.append(CandidatePrediction(sign=sign, confidence=conf))

            is_valid = dyn_result.confidence >= threshold and dyn_result.is_valid
            predicted_label = dyn_result.label if is_valid else "Unknown"

            return PredictionResponse(
                predicted_sign=predicted_label,
                confidence=float(dyn_result.confidence),
                is_recognized=is_valid,
                is_supported_vocabulary=dyn_result.label in self.dynamic_classes,
                model_name="SignBridge_Dynamic_BiLSTM",
                model_version="1.0",
                top_candidates=candidates,
                inference_duration_ms=round(duration_ms, 2),
            )
        except Exception as err:
            logger.error(f"Inference error in predict_sequence: {err}")
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return PredictionResponse(
                predicted_sign="Unknown",
                confidence=0.0,
                is_recognized=False,
                is_supported_vocabulary=False,
                model_name="SignBridge_Dynamic_BiLSTM",
                model_version="1.0",
                top_candidates=[],
                inference_duration_ms=round(duration_ms, 2),
            )


# Singleton service instance
recognition_service = RecognitionService()
