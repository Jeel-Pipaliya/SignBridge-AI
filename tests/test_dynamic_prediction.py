"""
SignBridge AI - Unit Tests for Dynamic Prediction Engine (Week 5)
Tests:
  - DynamicSignPredictor loading and inference
  - Confidence threshold gating
  - Minimum motion energy gating (filtering static poses)
  - Output schema structure
"""

import unittest
import numpy as np

from ai.dynamic.predict import DynamicSignPredictor, DynamicPredictionResult
import config


class TestDynamicPrediction(unittest.TestCase):
    """Test suite for DynamicSignPredictor."""

    def setUp(self):
        self.predictor = DynamicSignPredictor(
            model_path=config.DYNAMIC_MODEL_PATH,
            confidence_threshold=0.70,
            min_motion_energy=0.015,
        )

    def test_predictor_loaded(self):
        self.assertTrue(self.predictor.is_ready, "Trained Bi-LSTM checkpoint should be loaded.")
        self.assertEqual(len(self.predictor.classes), 3)

    def test_prediction_output_schema(self):
        # Create synthetic dynamic sequence with movement
        seq = np.zeros((30, 63), dtype=np.float32)
        for i in range(30):
            seq[i] = float(i) * 0.05

        result = self.predictor.predict_sequence(seq)
        self.assertIsInstance(result, DynamicPredictionResult)
        self.assertIn("label", result.to_dict())
        self.assertIn("confidence", result.to_dict())
        self.assertIn("motion_energy", result.to_dict())
        self.assertIn("all_probabilities", result.to_dict())

    def test_static_pose_rejected_by_motion_energy(self):
        # Static hold: zero motion
        static_seq = np.ones((30, 63), dtype=np.float32)
        result = self.predictor.predict_sequence(static_seq)
        # Should be gated by min_motion_energy
        self.assertFalse(result.is_valid)
        self.assertEqual(result.label, "...")


if __name__ == "__main__":
    unittest.main()
