"""
SignBridge AI - Unit Tests for Landmark Extraction (Week 2 & Week 4)
"""

import unittest
import numpy as np

from ai.preprocessing.landmark_extractor import (
    extract_landmarks_from_hand,
    extract_dual_hand_landmarks,
    extract_landmarks,
    extract_and_normalize,
    feature_summary,
    TOTAL_FEATURES,
    HAND_FEATURES,
    POSE_FEATURES,
)
from ai.preprocessing.normalize import normalize_landmarks


class DummyLandmark:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class DummyHand:
    def __init__(self):
        self.landmark = [DummyLandmark(i * 0.04, i * 0.03, i * 0.01) for i in range(21)]


class TestLandmarkExtractor(unittest.TestCase):
    """Test hand landmark extraction under multiple input modalities."""

    def test_total_features_constants(self):
        self.assertEqual(TOTAL_FEATURES, 225)
        self.assertEqual(HAND_FEATURES, 63)
        self.assertEqual(POSE_FEATURES, 99)

    def test_feature_summary(self):
        s = feature_summary()
        self.assertEqual(s["left_hand"]["features"], HAND_FEATURES)
        self.assertEqual(s["right_hand"]["features"], HAND_FEATURES)
        self.assertEqual(s["pose"]["features"], POSE_FEATURES)
        self.assertEqual(s["total"], TOTAL_FEATURES)

    def test_legacy_normalize_zeros(self):
        zero = np.zeros(TOTAL_FEATURES, dtype=np.float32)
        result = normalize_landmarks(zero)
        self.assertTrue(np.all(result == 0))

    def test_single_hand_extraction(self):
        dummy = DummyHand()
        raw = extract_landmarks_from_hand(dummy)
        self.assertEqual(raw.shape, (63,))
        self.assertEqual(raw.dtype, np.float32)
        self.assertAlmostEqual(raw[0], 0.0)  # lm0 x
        self.assertAlmostEqual(raw[1], 0.0)  # lm0 y

    def test_zero_hand_handling(self):
        # Must not crash and must return zero-vector
        raw = extract_landmarks_from_hand(None)
        self.assertEqual(raw.shape, (63,))
        self.assertTrue(np.all(raw == 0.0))

    def test_dual_hand_extraction(self):
        hand1 = DummyHand()
        hand2 = DummyHand()
        dual = extract_dual_hand_landmarks([hand1, hand2])
        self.assertEqual(dual.shape, (126,))

        # One hand missing in dual setup
        partial = extract_dual_hand_landmarks([hand1])
        self.assertEqual(partial.shape, (126,))
        self.assertFalse(np.all(partial[:63] == 0.0))
        self.assertTrue(np.all(partial[63:] == 0.0))

    def test_extract_and_normalize(self):
        dummy = DummyHand()
        norm = extract_and_normalize(dummy)
        self.assertEqual(norm.shape, (63,))
        # Wrist at origin
        self.assertAlmostEqual(norm[0], 0.0, places=5)
        self.assertAlmostEqual(norm[1], 0.0, places=5)
        self.assertAlmostEqual(norm[2], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
