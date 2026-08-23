"""
DAY 10 -- SignBridge AI  Week 2
Landmark Extractor: converts a single MediaPipe hand into a 63-feature vector.

Feature layout per hand (63 values):
    Landmark 0  -> x0, y0, z0
    Landmark 1  -> x1, y1, z1
    ...
    Landmark 20 -> x20, y20, z20

After extraction, landmarks are normalized relative to the wrist (landmark 0)
so that the model is robust to hand position within the frame.
"""

import numpy as np


# ---- Constants ---------------------------------------------------------------
NUM_LANDMARKS: int = 21          # MediaPipe Hands always gives 21 landmarks
NUM_FEATURES:  int = NUM_LANDMARKS * 3   # 63


# ---- Core functions ----------------------------------------------------------

def extract_landmarks(hand_landmarks) -> np.ndarray:
    """
    Convert a MediaPipe hand landmark object into a flat (63,) float32 array.

    Args:
        hand_landmarks: A single element from results.multi_hand_landmarks.

    Returns:
        np.ndarray of shape (63,), dtype float32.
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks relative to the wrist (landmark 0).

    Steps:
      1. Subtract wrist position -> makes coords relative to wrist.
      2. Scale so the maximum absolute value in the vector = 1.0
         -> makes the vector scale-invariant (robust to camera distance).

    Args:
        landmarks: np.ndarray of shape (63,), dtype float32.

    Returns:
        Normalized np.ndarray of shape (63,), dtype float32.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32).copy()

    # Reshape to (21, 3) for easy manipulation
    coords = landmarks.reshape(-1, 3)

    # Shift: subtract wrist position
    wrist = coords[0].copy()
    coords -= wrist

    # Scale
    flat = coords.flatten()
    max_abs = np.max(np.abs(flat))
    if max_abs > 1e-6:
        flat /= max_abs

    return flat.astype(np.float32)


def extract_and_normalize(hand_landmarks) -> np.ndarray:
    """Convenience: extract + normalize in one call."""
    raw = extract_landmarks(hand_landmarks)
    return normalize_landmarks(raw)


# ---- Standalone test ---------------------------------------------------------

if __name__ == "__main__":
    print("Landmark Extractor self-test")

    # Simulate 21 landmarks (random)
    class FakeLM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class FakeHand:
        landmark = [FakeLM(i * 0.05, i * 0.03, i * 0.01) for i in range(21)]

    raw = extract_landmarks(FakeHand())
    print(f"  Raw shape    : {raw.shape}")   # (63,)

    norm = normalize_landmarks(raw)
    print(f"  Norm shape   : {norm.shape}")  # (63,)
    print(f"  Norm max_abs : {np.max(np.abs(norm)):.4f}")  # <= 1.0
    print("  PASS")
