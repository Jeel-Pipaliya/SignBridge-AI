"""
SignBridge AI - Hand Landmark Extractor Module (Week 4 Phase 2)
Extracts raw and normalized coordinates from MediaPipe hand detection results.
Supports single-hand (63D), dual-hand (126D), and holistic inputs with safe zero-handling.
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np

from ai.preprocessing.normalization import normalize_landmarks

NUM_LANDMARKS_PER_HAND: int = 21
COORDS_PER_LANDMARK: int = 3
SINGLE_HAND_FEATURES: int = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
DUAL_HAND_FEATURES: int = SINGLE_HAND_FEATURES * 2                       # 126

# Backwards compatibility constants
NUM_HAND_LANDMARKS: int = 21
NUM_POSE_LANDMARKS: int = 33
HAND_FEATURES: int = SINGLE_HAND_FEATURES
POSE_FEATURES: int = NUM_POSE_LANDMARKS * 3  # 99
TOTAL_FEATURES: int = HAND_FEATURES * 2 + POSE_FEATURES  # 225


def feature_summary() -> dict:
    """Return a human-readable breakdown of the feature vector."""
    return {
        "left_hand":   {"landmarks": NUM_HAND_LANDMARKS, "features": HAND_FEATURES},
        "right_hand":  {"landmarks": NUM_HAND_LANDMARKS, "features": HAND_FEATURES},
        "pose":        {"landmarks": NUM_POSE_LANDMARKS, "features": POSE_FEATURES},
        "total":       TOTAL_FEATURES,
    }


def extract_landmarks_from_hand(hand_landmarks: Any) -> np.ndarray:
    """
    Extracts raw (x, y, z) coordinates from a single MediaPipe NormalizedLandmarkList.
    Returns: np.ndarray of shape (63,), dtype float32.
    """
    if hand_landmarks is None:
        return np.zeros(SINGLE_HAND_FEATURES, dtype=np.float32)

    coords: List[float] = []
    # If it's a MediaPipe NormalizedLandmarkList with .landmark attribute
    if hasattr(hand_landmarks, "landmark"):
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
    # If it's already an iterable/list of 3D points
    elif isinstance(hand_landmarks, (list, tuple, np.ndarray)):
        arr = np.asarray(hand_landmarks, dtype=np.float32).flatten()
        if arr.size == SINGLE_HAND_FEATURES:
            return arr
        elif arr.size == 0:
            return np.zeros(SINGLE_HAND_FEATURES, dtype=np.float32)
        else:
            coords = list(arr[:SINGLE_HAND_FEATURES])
            while len(coords) < SINGLE_HAND_FEATURES:
                coords.append(0.0)

    return np.array(coords, dtype=np.float32)


def extract_dual_hand_landmarks(multi_hand_landmarks: Optional[List[Any]]) -> np.ndarray:
    """
    Extracts 126-dimensional vector for two hands: [Hand 1 (63D), Hand 2 (63D)].
    If fewer than 2 hands are detected, remaining slots are zero-filled.
    """
    vector = np.zeros(DUAL_HAND_FEATURES, dtype=np.float32)
    if not multi_hand_landmarks:
        return vector

    # Hand 1
    if len(multi_hand_landmarks) >= 1 and multi_hand_landmarks[0] is not None:
        vector[:SINGLE_HAND_FEATURES] = extract_landmarks_from_hand(multi_hand_landmarks[0])

    # Hand 2
    if len(multi_hand_landmarks) >= 2 and multi_hand_landmarks[1] is not None:
        vector[SINGLE_HAND_FEATURES:] = extract_landmarks_from_hand(multi_hand_landmarks[1])

    return vector


def extract_landmarks(results: Any) -> np.ndarray:
    """
    Universal landmark extraction adapter:
    Handles MediaPipe Hands results (.multi_hand_landmarks) or MediaPipe Holistic results.
    """
    if results is None:
        return np.zeros(SINGLE_HAND_FEATURES, dtype=np.float32)

    # 1. MediaPipe Hands results object
    if hasattr(results, "multi_hand_landmarks"):
        hands = results.multi_hand_landmarks
        if hands and len(hands) > 0:
            return extract_landmarks_from_hand(hands[0])
        return np.zeros(SINGLE_HAND_FEATURES, dtype=np.float32)

    # 2. MediaPipe Holistic results object
    if hasattr(results, "right_hand_landmarks") or hasattr(results, "left_hand_landmarks"):
        right = extract_landmarks_from_hand(getattr(results, "right_hand_landmarks", None))
        left = extract_landmarks_from_hand(getattr(results, "left_hand_landmarks", None))
        # Default to whichever hand is non-zero
        if np.any(right):
            return right
        elif np.any(left):
            return left
        return np.zeros(SINGLE_HAND_FEATURES, dtype=np.float32)

    # 3. Direct landmark list
    return extract_landmarks_from_hand(results)


def extract_and_normalize(hand_landmarks: Any) -> np.ndarray:
    """
    Convenience function: extracts raw coordinates and normalizes them.
    If no landmarks are detected, returns zero-filled vector.
    """
    raw = extract_landmarks_from_hand(hand_landmarks)
    if not np.any(raw):
        return np.zeros(SINGLE_HAND_FEATURES, dtype=np.float32)
    return normalize_landmarks(raw)
