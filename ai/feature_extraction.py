"""
SignBridge AI - Feature Extraction Module (Week 1 Day 5)
Converts 21 MediaPipe hand landmarks (x, y, z) into a 63-dimensional normalized feature vector
that is invariant to translation (hand position in frame) and scale (distance from camera).
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Landmark index constants
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9  # Base of middle finger, stable anatomical reference for hand scale


def get_feature_names() -> List[str]:
    """
    Generate standard column headers for the 63 features:
    ['lm0_x', 'lm0_y', 'lm0_z', ..., 'lm20_x', 'lm20_y', 'lm20_z']
    """
    names = []
    for i in range(config.NUM_LANDMARKS):
        names.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])
    return names


def normalize_landmarks(
    landmarks: Union[List[Tuple[float, float, float]], np.ndarray],
    scale_method: str = "middle_mcp",
) -> np.ndarray:
    """
    Transforms 21 raw (x, y, z) landmarks into an invariant 63D feature vector.

    Steps:
      1. Translation Invariance (Wrist centering):
         Subtract the wrist coordinate (landmark 0) from all 21 landmarks.
         This shifts the coordinate origin (0, 0, 0) to the wrist, making
         features independent of where the hand appears on screen.

      2. Scale Invariance (Distance normalization):
         Compute a reference hand-size metric and divide all coordinates by it:
         - "middle_mcp": Distance from wrist (0) to middle finger base (9).
         - "max_distance": Maximum Euclidean distance from wrist across all 21 joints.
         This eliminates sensitivity to how close or far the user sits from the camera.

    Args:
        landmarks: List or array of 21 3D points [[x, y, z], ...].
        scale_method: Method for scale normalization ('middle_mcp' or 'max_distance').

    Returns:
        np.ndarray: Flattened 1D array of length 63 with float32 dtype.
    """
    pts = np.array(landmarks, dtype=np.float32)

    if pts.shape != (config.NUM_LANDMARKS, config.NUM_COORDINATES):
        raise ValueError(
            f"Expected landmark shape ({config.NUM_LANDMARKS}, {config.NUM_COORDINATES}), "
            f"got {pts.shape}"
        )

    # 1. Translation: center relative to wrist (index 0)
    wrist = pts[WRIST_INDEX].copy()
    centered = pts - wrist

    # 2. Scale normalization:
    if scale_method == "middle_mcp":
        # Distance from wrist (0) to middle finger MCP (9)
        ref_distance = np.linalg.norm(centered[MIDDLE_MCP_INDEX])
    elif scale_method == "max_distance":
        distances = np.linalg.norm(centered, axis=1)
        ref_distance = np.max(distances)
    else:
        raise ValueError(f"Unknown scale_method: {scale_method}")

    # Guard against division by zero (e.g. if all points are identical)
    if ref_distance < 1e-6:
        # Fall back to max distance if middle_mcp was degenerate
        ref_distance = np.max(np.linalg.norm(centered, axis=1))
        if ref_distance < 1e-6:
            ref_distance = 1.0

    normalized = centered / ref_distance

    # Flatten into 1D 63-element vector
    features = normalized.flatten()
    return features


def extract_features(
    landmarks: Optional[Union[List[Tuple[float, float, float]], np.ndarray]],
) -> Optional[np.ndarray]:
    """
    Safely extract 63D normalized features. Returns None if landmarks are None or empty.
    """
    if landmarks is None or len(landmarks) == 0:
        return None
    try:
        return normalize_landmarks(landmarks)
    except Exception as exc:
        print(f"[Warning] Feature extraction failed: {exc}")
        return None


def verify_invariance() -> bool:
    """
    Mathematical verification of translation and scale invariance.
    """
    print("-" * 60)
    print("Testing Feature Extraction Mathematical Invariance...")

    # Create dummy hand landmarks (21 points)
    np.random.seed(42)
    base_hand = np.random.uniform(0.3, 0.7, size=(21, 3)).astype(np.float32)
    feat_original = normalize_landmarks(base_hand)

    # 1. Test Translation Invariance (Shift hand by dx, dy, dz)
    shift_vector = np.array([0.25, -0.15, 0.40], dtype=np.float32)
    translated_hand = base_hand + shift_vector
    feat_translated = normalize_landmarks(translated_hand)

    diff_trans = np.max(np.abs(feat_original - feat_translated))
    trans_passed = diff_trans < 1e-5
    print(f"  [1] Translation Invariance: {'PASSED' if trans_passed else 'FAILED'} (Max Diff: {diff_trans:.2e})")

    # 2. Test Scale Invariance (Multiply hand coordinates by scale factor)
    # Hand scaled by 2.5x away from wrist
    wrist = base_hand[0]
    scaled_hand = wrist + (base_hand - wrist) * 2.5
    feat_scaled = normalize_landmarks(scaled_hand)

    diff_scale = np.max(np.abs(feat_original - feat_scaled))
    scale_passed = diff_scale < 1e-5
    print(f"  [2] Scale Invariance:       {'PASSED' if scale_passed else 'FAILED'} (Max Diff: {diff_scale:.2e})")

    # 3. Test Combined Translation + Scale
    combo_hand = (wrist + (base_hand - wrist) * 0.6) + np.array([-0.3, 0.5, 0.1], dtype=np.float32)
    feat_combo = normalize_landmarks(combo_hand)
    diff_combo = np.max(np.abs(feat_original - feat_combo))
    combo_passed = diff_combo < 1e-5
    print(f"  [3] Combined Invariance:    {'PASSED' if combo_passed else 'FAILED'} (Max Diff: {diff_combo:.2e})")

    overall = trans_passed and scale_passed and combo_passed
    print("-" * 60)
    return overall


if __name__ == "__main__":
    success = verify_invariance()
    if success:
        print("Feature Extraction Engine verified successfully!")
        names = get_feature_names()
        print(f"Total features: {len(names)} -> [{names[0]}, {names[1]}, ..., {names[-1]}]")
    else:
        sys.exit(1)
