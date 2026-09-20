"""
SignBridge AI - Unit Tests for Preprocessing & Normalization (Week 4)
"""

import unittest
import numpy as np

from ai.preprocessing.normalization import normalize_landmarks, normalize_coordinates, extract_hand_features


class TestPreprocessing(unittest.TestCase):
    """Test invariant coordinate normalization and edge cases."""

    def test_correct_input_shape_and_output(self):
        # 21 points with 3 coordinates = 63
        landmarks = np.random.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
        feat = normalize_landmarks(landmarks)
        self.assertEqual(feat.shape, (63,))
        self.assertEqual(feat.dtype, np.float32)

    def test_wrist_at_origin_after_normalization(self):
        landmarks = np.random.uniform(0.2, 0.8, size=(21, 3)).astype(np.float32)
        feat = normalize_landmarks(landmarks)
        coords = feat.reshape(21, 3)
        # Wrist is index 0
        np.testing.assert_allclose(coords[0], np.zeros(3), atol=1e-6)

    def test_translation_invariance(self):
        base = np.random.uniform(0.3, 0.7, size=(21, 3)).astype(np.float32)
        shift = np.array([0.4, -0.3, 0.5], dtype=np.float32)
        shifted = base + shift

        feat_base = normalize_landmarks(base)
        feat_shifted = normalize_landmarks(shifted)
        max_diff = np.max(np.abs(feat_base - feat_shifted))
        self.assertLess(max_diff, 1e-5)

    def test_scale_invariance(self):
        base = np.random.uniform(0.3, 0.7, size=(21, 3)).astype(np.float32)
        wrist = base[0]
        scaled = wrist + (base - wrist) * 3.0

        feat_base = normalize_landmarks(base)
        feat_scaled = normalize_landmarks(scaled)
        max_diff = np.max(np.abs(feat_base - feat_scaled))
        self.assertLess(max_diff, 1e-5)

    def test_invalid_input_shape(self):
        invalid = np.zeros((10, 3))
        with self.assertRaises(ValueError):
            normalize_landmarks(invalid)

    def test_safe_feature_extraction(self):
        self.assertIsNone(extract_hand_features(None))
        self.assertIsNone(extract_hand_features([]))


if __name__ == "__main__":
    unittest.main()
