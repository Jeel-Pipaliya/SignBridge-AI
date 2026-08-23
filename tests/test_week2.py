"""
Week 2 -- SignBridge AI
Unit tests for backend landmark extractor and prediction smoother.

Run:
    python tests/test_week2.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.preprocessing.landmark_extractor import (
    extract_landmarks,
    normalize_landmarks,
    extract_and_normalize,
    NUM_FEATURES,
    NUM_LANDMARKS,
)
from backend.recognition.realtime_recognition import PredictionSmoother


# ---- Helpers -----------------------------------------------------------------

def _pass(name: str) -> None:
    print(f"  [OK]  {name}")


def _fail(name: str, reason: str) -> None:
    print(f"  [!!]  {name} -- {reason}")
    sys.exit(1)


# ---- Fake hand landmark -------------------------------------------------------

class _FakeLM:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeHand:
    """Simulates a MediaPipe hand landmark object."""
    def __init__(self, offset=0.0):
        self.landmark = [
            _FakeLM(i * 0.05 + offset, i * 0.03, i * 0.01)
            for i in range(NUM_LANDMARKS)
        ]


# ---- Extractor tests ---------------------------------------------------------

def test_extract_shape():
    hand = _FakeHand()
    raw = extract_landmarks(hand)
    assert raw.shape == (NUM_FEATURES,), f"Wrong shape: {raw.shape}"
    _pass(f"extract_landmarks() shape == ({NUM_FEATURES},)")


def test_extract_dtype():
    raw = extract_landmarks(_FakeHand())
    assert raw.dtype == np.float32, f"Wrong dtype: {raw.dtype}"
    _pass("extract_landmarks() dtype is float32")


def test_normalize_shape():
    raw = extract_landmarks(_FakeHand())
    norm = normalize_landmarks(raw)
    assert norm.shape == (NUM_FEATURES,), f"Wrong shape: {norm.shape}"
    _pass(f"normalize_landmarks() output shape == ({NUM_FEATURES},)")


def test_normalize_range():
    raw  = extract_landmarks(_FakeHand())
    norm = normalize_landmarks(raw)
    max_abs = float(np.max(np.abs(norm)))
    assert max_abs <= 1.0 + 1e-5, f"Out of range: {max_abs}"
    _pass(f"normalize_landmarks() values in [-1, 1]  (max={max_abs:.4f})")


def test_normalize_wrist_at_origin():
    """After normalization the wrist (landmark 0) should be at (0, 0, 0)."""
    raw  = extract_landmarks(_FakeHand(offset=0.5))
    norm = normalize_landmarks(raw)
    coords = norm.reshape(-1, 3)
    wrist = coords[0]
    assert np.allclose(wrist, [0, 0, 0], atol=1e-6), \
        f"Wrist not at origin: {wrist}"
    _pass("Wrist landmark is at origin after normalization")


def test_normalize_zeros():
    zero = np.zeros(NUM_FEATURES, dtype=np.float32)
    norm = normalize_landmarks(zero)
    assert np.all(norm == 0), "Zero input should stay zero"
    _pass("All-zero input stays all-zero after normalization")


def test_extract_and_normalize_convenience():
    result = extract_and_normalize(_FakeHand())
    assert result.shape == (NUM_FEATURES,)
    assert result.dtype == np.float32
    _pass("extract_and_normalize() returns correct shape + dtype")


def test_position_invariance():
    """
    Two hands at different positions but same shape should give
    the same normalized vector.
    """
    hand_a = _FakeHand(offset=0.0)
    hand_b = _FakeHand(offset=0.8)   # shifted
    norm_a = extract_and_normalize(hand_a)
    norm_b = extract_and_normalize(hand_b)
    assert np.allclose(norm_a, norm_b, atol=1e-5), \
        "Normalization is not position-invariant"
    _pass("Normalization is position-invariant")


# ---- Smoother tests ----------------------------------------------------------

def test_smoother_majority_vote():
    s = PredictionSmoother(window=5)
    for _ in range(3):
        s.update("hello")
    s.update("yes")
    result = s.update("no")
    assert result == "hello", f"Expected 'hello', got '{result}'"
    _pass("PredictionSmoother returns majority vote")


def test_smoother_window_size():
    s = PredictionSmoother(window=3)
    s.update("hello")
    s.update("hello")
    s.update("hello")
    # now push out old 'hello' entries
    s.update("stop")
    s.update("stop")
    result = s.update("stop")
    assert result == "stop", f"Window didn't slide: got '{result}'"
    _pass("PredictionSmoother sliding window works")


def test_smoother_reset():
    s = PredictionSmoother(window=5)
    for _ in range(5):
        s.update("hello")
    s.reset()
    result = s.update("stop")
    assert result == "stop", f"Expected 'stop' after reset, got '{result}'"
    _pass("PredictionSmoother reset() clears history")


# ---- Runner ------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  SignBridge AI -- Week 2 Tests")
    print("=" * 55)

    test_extract_shape()
    test_extract_dtype()
    test_normalize_shape()
    test_normalize_range()
    test_normalize_wrist_at_origin()
    test_normalize_zeros()
    test_extract_and_normalize_convenience()
    test_position_invariance()

    print()
    test_smoother_majority_vote()
    test_smoother_window_size()
    test_smoother_reset()

    print()
    print("  All Week 2 tests passed!")
    print("=" * 55)


if __name__ == "__main__":
    main()
