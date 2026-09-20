"""
SignBridge AI - Landmark Normalization Module (Week 4 Phase 2)
Provides translation and scale invariance for 21 3D hand landmarks.

Mathematical Formulation:
  1. Translation Invariance (Wrist Centering):
     p'_i = p_i - p_wrist   for all i in [0..20]
     Moves wrist to coordinate origin (0, 0, 0).

  2. Scale Invariance (Distance Normalization):
     d_ref = ||p'_middle_mcp||_2 (or max ||p'_i||_2)
     p''_i = p'_i / d_ref
     Removes camera proximity and hand-size variance.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

WRIST_INDEX: int = 0
MIDDLE_MCP_INDEX: int = 9  # Base of middle finger, stable anatomical anchor
NUM_LANDMARKS: int = 21
NUM_COORDS: int = 3


def normalize_landmarks(
    landmarks: Union[List[Tuple[float, float, float]], np.ndarray],
    scale_method: str = "middle_mcp",
) -> np.ndarray:
    """
    Transforms 21 raw (x, y, z) landmarks into an invariant 63-dimensional feature vector.

    Args:
        landmarks: List or array of shape (21, 3) or flat array of length 63.
        scale_method: 'middle_mcp' (anatomical anchor) or 'max_distance' (bounding radius).

    Returns:
        Flattened 1D np.ndarray of length 63, dtype float32.
    """
    pts = np.asarray(landmarks, dtype=np.float32)

    if pts.size != NUM_LANDMARKS * NUM_COORDS:
        raise ValueError(
            f"Expected {NUM_LANDMARKS * NUM_COORDS} values (21 landmarks x 3 coords), "
            f"got size {pts.size}"
        )

    pts = pts.reshape(NUM_LANDMARKS, NUM_COORDS)

    # 1. Translation Invariance: Shift origin to wrist
    wrist = pts[WRIST_INDEX].copy()
    centered = pts - wrist

    # 2. Scale Invariance: Divide by reference anatomical distance
    if scale_method == "middle_mcp":
        ref_distance = float(np.linalg.norm(centered[MIDDLE_MCP_INDEX]))
    elif scale_method == "max_distance":
        ref_distance = float(np.max(np.linalg.norm(centered, axis=1)))
    else:
        raise ValueError(f"Unsupported scale_method: '{scale_method}'")

    # Guard against division by zero (e.g. all points degenerate or identical)
    if ref_distance < 1e-6:
        ref_distance = float(np.max(np.linalg.norm(centered, axis=1)))
        if ref_distance < 1e-6:
            ref_distance = 1.0

    normalized = centered / ref_distance
    return normalized.flatten().astype(np.float32)


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Convenience alias for normalize_landmarks.
    """
    return normalize_landmarks(coords)


def extract_hand_features(landmarks: Optional[Union[List, np.ndarray]]) -> Optional[np.ndarray]:
    """
    Safely extract 63D normalized features. Returns None if input is invalid or empty.
    """
    if landmarks is None or len(landmarks) == 0:
        return None
    try:
        return normalize_landmarks(landmarks)
    except Exception:
        return None
