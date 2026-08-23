"""
DAY 5 — SignBridge AI
Landmark Normalization.

Week 1 goal: understand WHY we normalize and implement a basic strategy.

Strategy
────────
We normalize each landmark section relative to its own reference point:
  - Hands  → relative to wrist (landmark 0)
  - Pose   → relative to the midpoint between left & right hips
             (landmarks 23 and 24 in MediaPipe Pose)

After shifting, we scale so the maximum absolute value in the section = 1.
This makes features roughly camera-distance-independent.

NOTE: This is a reasonable Week 1 baseline. A more sophisticated approach
(e.g., Procrustes alignment) will be introduced in Week 2.
"""

import numpy as np
from ai.preprocessing.landmark_extractor import (
    HAND_FEATURES,
    POSE_FEATURES,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
)


# ─── Index helpers ────────────────────────────────────────────────────────────
# In the flat landmark vector the layout is:
#   [0   : 63 ) → left hand  (21 × 3)
#   [63  : 126) → right hand (21 × 3)
#   [126 : 225) → pose       (33 × 3)

_LEFT_START  = 0
_RIGHT_START = HAND_FEATURES          # 63
_POSE_START  = HAND_FEATURES * 2      # 126

# Wrist is landmark index 0 → first 3 values of each hand section
_WRIST_SLICE = slice(0, 3)

# Pose landmarks 23 (left hip) and 24 (right hip) — indices in pose section
_LEFT_HIP_IDX  = 23
_RIGHT_HIP_IDX = 24


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _normalize_section(section: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Shift all (x, y, z) triples in `section` by `reference`,
    then scale to [-1, 1].

    Args:
        section:   1-D array of length N*3.
        reference: 1-D array of length 3 (the anchor point to subtract).

    Returns:
        Normalized array of same length and dtype.
    """
    # Reshape to (N, 3), subtract reference, flatten back
    coords = section.reshape(-1, 3)
    coords = coords - reference         # broadcast subtract

    flat = coords.flatten()

    # Scale
    max_abs = np.max(np.abs(flat))
    if max_abs > 1e-6:
        flat = flat / max_abs

    return flat.astype(np.float32)


# ─── Public API ───────────────────────────────────────────────────────────────

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize a (225,) landmark vector produced by extract_landmarks().

    Each body-part section is independently shifted to its reference point
    and then scaled so that the maximum absolute value = 1.0.

    Sections that are all-zeros (i.e. the body part was not detected)
    are left as zeros — no normalization is applied.

    Args:
        landmarks: np.ndarray of shape (225,), dtype float32.

    Returns:
        Normalized np.ndarray of shape (225,), dtype float32.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32).copy()

    # ── Left hand ─────────────────────────────────────────────────────────────
    left = landmarks[_LEFT_START : _LEFT_START + HAND_FEATURES]
    if np.any(left != 0):
        wrist_l = left[:3].copy()
        landmarks[_LEFT_START : _LEFT_START + HAND_FEATURES] = (
            _normalize_section(left, wrist_l)
        )

    # ── Right hand ────────────────────────────────────────────────────────────
    right = landmarks[_RIGHT_START : _RIGHT_START + HAND_FEATURES]
    if np.any(right != 0):
        wrist_r = right[:3].copy()
        landmarks[_RIGHT_START : _RIGHT_START + HAND_FEATURES] = (
            _normalize_section(right, wrist_r)
        )

    # ── Pose ──────────────────────────────────────────────────────────────────
    pose = landmarks[_POSE_START : _POSE_START + POSE_FEATURES]
    if np.any(pose != 0):
        pose_coords = pose.reshape(-1, 3)
        left_hip  = pose_coords[_LEFT_HIP_IDX].copy()
        right_hip = pose_coords[_RIGHT_HIP_IDX].copy()
        hip_mid   = (left_hip + right_hip) / 2.0
        landmarks[_POSE_START : _POSE_START + POSE_FEATURES] = (
            _normalize_section(pose, hip_mid)
        )

    return landmarks
