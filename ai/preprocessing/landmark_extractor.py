"""
DAY 5 — SignBridge AI
Landmark Extractor: converts a MediaPipe Holistic result into a flat NumPy vector.

Feature layout (225 values total):
  Left  Hand → 21 landmarks × 3 (x, y, z) =  63 values
  Right Hand → 21 landmarks × 3 (x, y, z) =  63 values
  Pose       → 33 landmarks × 3 (x, y, z) =  99 values
  ─────────────────────────────────────────
  Total                                     = 225 values

Missing detections are filled with zeros so the vector always has fixed length.
"""

import numpy as np
from typing import Any


# ─── Constants ────────────────────────────────────────────────────────────────
NUM_HAND_LANDMARKS: int = 21      # MediaPipe Hands gives 21 landmarks/hand
NUM_POSE_LANDMARKS: int = 33      # MediaPipe Pose gives 33 landmarks
HAND_FEATURES: int = NUM_HAND_LANDMARKS * 3   # 63
POSE_FEATURES: int = NUM_POSE_LANDMARKS * 3   # 99
TOTAL_FEATURES: int = HAND_FEATURES * 2 + POSE_FEATURES  # 225


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _hand_to_array(hand_landmarks) -> np.ndarray:
    """Convert a hand LandmarkList to a (63,) float32 array."""
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data, dtype=np.float32)


def _pose_to_array(pose_landmarks) -> np.ndarray:
    """
    Convert pose LandmarkList to a (99,) float32 array.
    Uses only x, y, z — visibility is intentionally excluded to keep
    the feature vector consistent regardless of detection confidence.
    """
    data = []
    for lm in pose_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data, dtype=np.float32)


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_landmarks(results: Any) -> np.ndarray:
    """
    Extract a fixed-length (225,) landmark vector from a MediaPipe
    Holistic result object.

    Args:
        results: The return value of holistic.process(rgb_frame).

    Returns:
        np.ndarray of shape (225,) and dtype float32.
        Sections with no detected body part are zero-filled.
    """
    # Left hand (63 values)
    if results.left_hand_landmarks:
        left_hand = _hand_to_array(results.left_hand_landmarks)
    else:
        left_hand = np.zeros(HAND_FEATURES, dtype=np.float32)

    # Right hand (63 values)
    if results.right_hand_landmarks:
        right_hand = _hand_to_array(results.right_hand_landmarks)
    else:
        right_hand = np.zeros(HAND_FEATURES, dtype=np.float32)

    # Pose (99 values)
    if results.pose_landmarks:
        pose = _pose_to_array(results.pose_landmarks)
    else:
        pose = np.zeros(POSE_FEATURES, dtype=np.float32)

    return np.concatenate([left_hand, right_hand, pose])


def feature_summary() -> dict:
    """Return a human-readable breakdown of the feature vector."""
    return {
        "left_hand":   {"landmarks": NUM_HAND_LANDMARKS, "features": HAND_FEATURES},
        "right_hand":  {"landmarks": NUM_HAND_LANDMARKS, "features": HAND_FEATURES},
        "pose":        {"landmarks": NUM_POSE_LANDMARKS, "features": POSE_FEATURES},
        "total":       TOTAL_FEATURES,
    }
