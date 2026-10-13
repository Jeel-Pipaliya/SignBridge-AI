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
        self._class_centroids: Dict[str, np.ndarray] = {}
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Load static and dynamic models from configured paths."""
        # 1. Static ISL Predictor
        static_model_path = PROJECT_ROOT / settings.STATIC_MODEL_PATH
        if not static_model_path.exists():
            static_model_path = PROJECT_ROOT / "ai" / "models" / "isl_classifier.pkl"
        if not static_model_path.exists():
            static_model_path = PROJECT_ROOT / "models" / "isl_classifier.pkl"

        encoder_path = PROJECT_ROOT / "ai" / "models" / "label_encoder.pkl"
        if not encoder_path.exists():
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

        # 4. Load Canonical Landmark Centroids for Real Geometric Tutor Similarity
        self._load_canonical_centroids()

    def _load_canonical_centroids(self) -> None:
        """Computes unit-normalized class prototypes from training landmarks for tutor similarity."""
        train_feat_path = PROJECT_ROOT / "ai" / "data" / "landmarks" / "train" / "features.npy"
        train_labels_path = PROJECT_ROOT / "ai" / "data" / "landmarks" / "train" / "labels.npy"
        self._class_prototypes: Dict[str, List[np.ndarray]] = {}

        if train_feat_path.exists() and train_labels_path.exists():
            try:
                feats = np.load(train_feat_path)
                labels = np.load(train_labels_path, allow_pickle=True)
                for lbl in np.unique(labels):
                    mask = (labels == lbl)
                    sub = feats[mask]
                    if len(sub) == 0:
                        continue

                    lbl_key = str(lbl).strip().upper()
                    # 1. Mean centroid
                    c = np.mean(sub, axis=0)
                    norm = np.linalg.norm(c)
                    exemplars = [c / norm if norm > 1e-9 else c]

                    # 2. Add top medoids
                    if len(sub) > 1:
                        sub_norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9
                        normed_sub = sub / sub_norms
                        sim_matrix = np.dot(normed_sub, normed_sub.T)
                        avg_sims = np.mean(sim_matrix, axis=1)
                        best_indices = np.argsort(avg_sims)[::-1][:min(3, len(sub))]
                        for idx in best_indices:
                            exemplars.append(normed_sub[idx])

                    self._class_prototypes[lbl_key] = exemplars
                    self._class_centroids[lbl_key] = exemplars[0]

                logger.info(f"Initialized {len(self._class_prototypes)} canonical landmark prototype classes for Tutor mode")
            except Exception as err:
                logger.warning(f"Could not load landmark training features for prototypes: {err}")

    def compute_target_similarity(
        self,
        features: np.ndarray,
        target_sign: str,
        predicted_sign: str,
        class_probabilities: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, bool, str]:
        """
        Calculates geometric landmark cosine similarity against the canonical target gesture,
        blended with class classification probability and anatomical alignment feedback.
        """
        target_norm = target_sign.strip().upper()
        pred_norm = predicted_sign.strip().upper() if predicted_sign else ""
        class_probs = class_probabilities or {}

        # 1. Model classification probability for the target sign
        target_prob = float(class_probs.get(target_norm, class_probs.get(target_sign, 0.0)))

        # 2. Geometric alignment against canonical landmark prototypes (with mirror symmetry)
        geom_similarity = 0.0
        prototypes = getattr(self, "_class_prototypes", {}).get(target_norm)
        if not prototypes and target_norm in self._class_centroids:
            prototypes = [self._class_centroids[target_norm]]

        if prototypes:
            feat_norm = np.linalg.norm(features)
            if feat_norm > 1e-9:
                f_unit = features / feat_norm

                # Also generate mirrored vector (x -> -x) for mirrored webcams / opposite hand
                f_mirror = features.copy().reshape(-1, 3)
                f_mirror[:, 0] = -f_mirror[:, 0]
                f_mirror_unit = f_mirror.flatten() / feat_norm

                best_cos = -1.0
                for proto in prototypes:
                    cos1 = float(np.dot(f_unit, proto))
                    cos2 = float(np.dot(f_mirror_unit, proto))
                    best_cos = max(best_cos, cos1, cos2)

                # Scale cosine range [-0.10, 0.92] to intuitive [0.0, 1.0]
                geom_similarity = float(np.clip((best_cos - 0.10) / (0.90 - 0.10), 0.0, 1.0))
            else:
                geom_similarity = 0.0
        else:
            geom_similarity = target_prob

        # 3. Blended continuous similarity score
        is_exact_match = (pred_norm == target_norm)
        if is_exact_match:
            # Top class matched target
            score = max(geom_similarity, target_prob, 0.78)
        elif target_prob >= 0.25:
            # High probability candidate for target
            score = 0.60 * geom_similarity + 0.40 * target_prob
            if geom_similarity >= 0.70:
                score = max(score, 0.72)
        else:
            # Another sign predicted or hand forming gesture
            if geom_similarity >= 0.75:
                # Strong landmark pose match even if classifier flagged similar class
                score = max(0.68, geom_similarity * 0.90)
            else:
                score = min(geom_similarity * 0.85, 0.65)

        final_similarity = round(float(np.clip(score, 0.0, 0.98)), 4)
        is_match = bool(final_similarity >= 0.70 or (is_exact_match and target_prob >= 0.55))

        # 4. Generate anatomical, pedagogical feedback
        if is_match or final_similarity >= 0.72:
            feedback = f"Excellent! Your hand pose accurately matches '{target_sign}'."
        elif final_similarity >= 0.45:
            if pred_norm and pred_norm != target_norm and pred_norm not in ("UNKNOWN", "UNKNOWN / LOW CONFIDENCE"):
                feedback = f"Forming '{target_sign}': Currently leaning towards '{pred_norm}'. Adjust finger angles."
            else:
                feedback = f"Good attempt! Refine your finger extension and wrist alignment for '{target_sign}'."
        else:
            if pred_norm and pred_norm != target_norm and pred_norm not in ("UNKNOWN", "UNKNOWN / LOW CONFIDENCE"):
                feedback = f"Hand pose detected as '{pred_norm}'. Realign your fingers according to the '{target_sign}' guide."
            else:
                feedback = f"Hand pose does not yet match '{target_sign}'. Follow the anatomical illustration."

        return final_similarity, is_match, feedback

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
        target_sign: Optional[str] = None,
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
                target_sign=target_sign,
                target_similarity=0.0 if target_sign else None,
                is_target_match=False if target_sign else None,
                feedback_message="Static model unavailable." if target_sign else None,
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
            # Always output the real-world sign prediction without masking as 'Unknown'
            predicted_label = result.raw_label if (result.raw_label and result.raw_label not in ("Unknown", "Unknown / Low Confidence")) else result.label

            # Compute target similarity if practicing with a target sign
            target_sim = None
            is_match = None
            feedback = None
            if target_sign:
                target_sim, is_match, feedback = self.compute_target_similarity(
                    features=features,
                    target_sign=target_sign,
                    predicted_sign=predicted_label,
                    class_probabilities=result.all_probabilities,
                )

            return PredictionResponse(
                predicted_sign=predicted_label,
                confidence=float(result.confidence),
                is_recognized=True,
                is_supported_vocabulary=result.raw_label in self.supported_classes,
                model_name=self._static_metadata.get("model_name", "SignBridge_ISL_MLP_NeuralNet"),
                model_version=self._static_metadata.get("version", "1.0"),
                top_candidates=candidates,
                inference_duration_ms=round(duration_ms, 2),
                handedness=handedness,
                target_sign=target_sign,
                target_similarity=target_sim,
                is_target_match=is_match,
                feedback_message=feedback,
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
                target_sign=target_sign,
                target_similarity=0.0 if target_sign else None,
                is_target_match=False if target_sign else None,
                feedback_message="Could not analyze hand landmarks." if target_sign else None,
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
            predicted_label = getattr(dyn_result, "raw_label", dyn_result.label)
            if predicted_label in ("Unknown", "Unknown / Low Confidence") and dyn_result.all_probabilities:
                predicted_label = max(dyn_result.all_probabilities.items(), key=lambda x: x[1])[0]

            return PredictionResponse(
                predicted_sign=predicted_label,
                confidence=float(dyn_result.confidence),
                is_recognized=True,
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

    def predict_frame(
        self,
        image_data: str,
        confidence_threshold: Optional[float] = None,
        target_sign: Optional[str] = None,
    ) -> PredictionResponse:
        """
        Processes a base64 encoded video camera frame, runs MediaPipe HandDetector,
        extracts 21 3D landmarks, and performs invariant classification and target similarity.
        """
        start_time = time.perf_counter()
        threshold = confidence_threshold if confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD

        try:
            if not hasattr(self, "_hand_detector") or self._hand_detector is None:
                from ai.hand_detection import HandDetector
                self._hand_detector = HandDetector(static_image_mode=False)

            import base64
            import cv2

            if "," in image_data:
                _, b64_str = image_data.split(",", 1)
            else:
                b64_str = image_data

            img_bytes = base64.b64decode(b64_str)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode image frame")

            _, detected_hands = self._hand_detector.process(frame, draw=False)

            if not detected_hands:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return PredictionResponse(
                    predicted_sign="No Hand Detected",
                    confidence=0.0,
                    is_recognized=False,
                    is_supported_vocabulary=False,
                    model_name="SignBridge_ISL_MLP_NeuralNet",
                    model_version="1.0",
                    top_candidates=[],
                    inference_duration_ms=round(duration_ms, 2),
                    handedness=None,
                    hand_detected=False,
                    landmarks=[],
                    target_sign=target_sign,
                    target_similarity=0.0 if target_sign else None,
                    is_target_match=False if target_sign else None,
                    feedback_message=f"No hand detected. Position your hand clearly in camera to practice '{target_sign}'." if target_sign else None,
                )

            primary_hand = detected_hands[0]
            landmarks_list = [
                LandmarkPoint(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
                for pt in primary_hand.landmark_list
            ]

            pred_response = self.predict_landmarks(
                landmarks=landmarks_list,
                handedness=primary_hand.handedness,
                confidence_threshold=threshold,
                target_sign=target_sign,
            )
            pred_response.hand_detected = True
            pred_response.landmarks = landmarks_list
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            pred_response.inference_duration_ms = round(duration_ms, 2)
            return pred_response

        except Exception as err:
            logger.error(f"Error processing frame in predict_frame: {err}")
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return PredictionResponse(
                predicted_sign="Error",
                confidence=0.0,
                is_recognized=False,
                is_supported_vocabulary=False,
                model_name="SignBridge_ISL_MLP_NeuralNet",
                model_version="1.0",
                top_candidates=[],
                inference_duration_ms=round(duration_ms, 2),
                handedness=None,
                hand_detected=False,
                landmarks=[],
                target_sign=target_sign,
                target_similarity=0.0 if target_sign else None,
                is_target_match=False if target_sign else None,
                feedback_message="Could not process video frame." if target_sign else None,
            )


# Singleton service instance
recognition_service = RecognitionService()

