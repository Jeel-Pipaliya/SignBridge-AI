"""
DAY 5 -- SignBridge AI
Tests for the landmark extractor and normalization module.

Run:
    python tests/test_landmarks.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.preprocessing.landmark_extractor import (
    extract_landmarks,
    feature_summary,
    TOTAL_FEATURES,
    HAND_FEATURES,
    POSE_FEATURES,
)
from ai.preprocessing.normalize import normalize_landmarks


# ---- Helpers -----------------------------------------------------------------

def _pass(name: str) -> None:
    print(f"  [OK]  {name}")


def _fail(name: str, reason: str) -> None:
    print(f"  [!!]  {name} -- {reason}")
    sys.exit(1)


# ---- Tests -------------------------------------------------------------------

def test_total_features():
    """Total feature count must equal 225."""
    expected = 225
    if TOTAL_FEATURES == expected:
        _pass(f"TOTAL_FEATURES == {expected}")
    else:
        _fail("TOTAL_FEATURES", f"got {TOTAL_FEATURES}, expected {expected}")


def test_zero_vector_shape():
    """A vector of zeros must have shape (225,)."""
    vec = np.zeros(TOTAL_FEATURES, dtype=np.float32)
    assert vec.shape == (TOTAL_FEATURES,), f"Shape mismatch: {vec.shape}"
    _pass("Zero-vector shape (225,)")


def test_feature_summary():
    """feature_summary() must report correct numbers."""
    s = feature_summary()
    assert s["left_hand"]["features"]  == HAND_FEATURES, "Left hand features wrong"
    assert s["right_hand"]["features"] == HAND_FEATURES, "Right hand features wrong"
    assert s["pose"]["features"]       == POSE_FEATURES, "Pose features wrong"
    assert s["total"]                  == TOTAL_FEATURES, "Total features wrong"
    _pass("feature_summary() values consistent")


def test_normalization_range():
    """Normalized output values must be in [-1, 1]."""
    rng = np.random.default_rng(42)
    raw = rng.random(TOTAL_FEATURES).astype(np.float32)
    normalized = normalize_landmarks(raw)
    assert normalized.shape == (TOTAL_FEATURES,), "Shape changed after normalization"
    max_abs = float(np.max(np.abs(normalized)))
    assert max_abs <= 1.0 + 1e-5, f"Value out of range: max_abs = {max_abs}"
    _pass(f"Normalized values in [-1, 1]  (max_abs={max_abs:.4f})")


def test_normalize_zeros():
    """All-zero vector must remain all-zero after normalization."""
    zero = np.zeros(TOTAL_FEATURES, dtype=np.float32)
    result = normalize_landmarks(zero)
    assert np.all(result == 0), "Zero vector changed after normalization"
    _pass("All-zero input -> all-zero output")


def test_normalization_dtype():
    """Output dtype must be float32."""
    data = np.ones(TOTAL_FEATURES, dtype=np.float64)
    result = normalize_landmarks(data)
    assert result.dtype == np.float32, f"Wrong dtype: {result.dtype}"
    _pass("Normalization output dtype is float32")


# ---- Runner ------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  SignBridge AI -- Landmark Tests (Day 5)")
    print("=" * 50)

    test_total_features()
    test_zero_vector_shape()
    test_feature_summary()
    test_normalization_range()
    test_normalize_zeros()
    test_normalization_dtype()

    print()
    print("  All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
