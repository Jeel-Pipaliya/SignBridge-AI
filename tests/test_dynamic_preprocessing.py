"""
SignBridge AI - Unit Tests for Dynamic Sequence Preprocessing (Week 5)
Tests:
  - Fixed output shape (T=30, F=63)
  - Variable-length resampling (linear interpolation)
  - Edge and zero padding
  - Coordinate normalization integration
  - Motion energy computation
  - Sequence data augmentation
  - Invalid input handling
"""

import unittest
import numpy as np

from ai.dynamic.preprocessing import (
    resample_sequence,
    pad_or_truncate_sequence,
    preprocess_dynamic_sequence,
    augment_sequence,
    compute_sequence_motion_energy,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_FEATURE_DIM,
)


class TestDynamicPreprocessing(unittest.TestCase):
    """Test suite for dynamic sequence preprocessing algorithms."""

    def test_resample_sequence_short(self):
        # 10 frames -> 30 frames
        short_seq = np.random.randn(10, 63).astype(np.float32)
        resampled = resample_sequence(short_seq, target_length=30)
        self.assertEqual(resampled.shape, (30, 63))
        # First and last frames should match endpoints
        np.testing.assert_allclose(resampled[0], short_seq[0], atol=1e-5)
        np.testing.assert_allclose(resampled[-1], short_seq[-1], atol=1e-5)

    def test_resample_sequence_long(self):
        # 50 frames -> 30 frames
        long_seq = np.random.randn(50, 63).astype(np.float32)
        resampled = resample_sequence(long_seq, target_length=30)
        self.assertEqual(resampled.shape, (30, 63))

    def test_pad_or_truncate(self):
        # Edge padding
        seq = np.ones((12, 63), dtype=np.float32)
        padded = pad_or_truncate_sequence(seq, target_length=30, pad_mode="edge")
        self.assertEqual(padded.shape, (30, 63))
        np.testing.assert_allclose(padded[12:], np.ones((18, 63)), atol=1e-5)

        # Truncation
        long_seq = np.ones((45, 63), dtype=np.float32)
        truncated = pad_or_truncate_sequence(long_seq, target_length=30)
        self.assertEqual(truncated.shape, (30, 63))

    def test_preprocess_dynamic_sequence(self):
        # Valid list of 25 raw frames
        frames = [np.random.randn(63).astype(np.float32) for _ in range(25)]
        out = preprocess_dynamic_sequence(frames, target_length=30, feature_dim=63)
        self.assertEqual(out.shape, (30, 63))
        self.assertEqual(out.dtype, np.float32)

    def test_preprocess_empty_sequence_raises(self):
        with self.assertRaises(ValueError):
            preprocess_dynamic_sequence([])

    def test_preprocess_invalid_feature_dim_raises(self):
        invalid_frames = [np.random.randn(42) for _ in range(15)]
        with self.assertRaises(ValueError):
            preprocess_dynamic_sequence(invalid_frames, feature_dim=63)

    def test_augment_sequence(self):
        seq = np.random.randn(30, 63).astype(np.float32)
        rng = np.random.default_rng(123)
        aug = augment_sequence(seq, rng=rng)
        self.assertEqual(aug.shape, (30, 63))
        # Ensure values changed due to augmentation
        self.assertFalse(np.allclose(seq, aug))

    def test_motion_energy_static_vs_dynamic(self):
        # Static hold: identical frames across time -> 0 motion energy
        static_seq = np.repeat(np.random.randn(1, 63).astype(np.float32), 30, axis=0)
        energy_static = compute_sequence_motion_energy(static_seq)
        self.assertAlmostEqual(energy_static, 0.0, places=5)

        # Dynamic motion: frame-to-frame displacements
        dynamic_seq = np.zeros((30, 63), dtype=np.float32)
        for i in range(30):
            dynamic_seq[i, :] = float(i) * 0.1
        energy_dynamic = compute_sequence_motion_energy(dynamic_seq)
        self.assertGreater(energy_dynamic, 0.05)


if __name__ == "__main__":
    unittest.main()
