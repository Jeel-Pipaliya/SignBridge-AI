"""
SignBridge AI - Unit Tests for Prediction & Confidence Gating (Week 4)
"""

import unittest
import numpy as np

import config
from ai.predict import SignPredictor, PredictionResult


class TestPredictionPipeline(unittest.TestCase):
    """Test SignPredictor real-time predictions and thresholds."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = SignPredictor(confidence_threshold=0.70)

    def test_prediction_output_structure(self):
        dummy_feat = np.random.uniform(-1.0, 1.0, size=(63,)).astype(np.float32)
        res = self.predictor.predict_features(dummy_feat)

        self.assertIsInstance(res, PredictionResult)
        self.assertIsInstance(res.label, str)
        self.assertIsInstance(res.raw_label, str)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)
        self.assertIn(res.raw_label, self.predictor.classes)

    def test_confidence_threshold_gating(self):
        # Predict with a very high threshold (0.999) - should mark as low confidence/unknown
        strict_predictor = SignPredictor(confidence_threshold=0.999)
        dummy_feat = np.random.uniform(-0.5, 0.5, size=(63,)).astype(np.float32)
        res = strict_predictor.predict_features(dummy_feat)

        if res.confidence < 0.999:
            self.assertFalse(res.is_valid)
            self.assertEqual(res.label, "Unknown / Low Confidence")

    def test_no_hand_input(self):
        res = self.predictor.predict_landmarks(None)
        self.assertEqual(res.label, "No Hand")
        self.assertEqual(res.confidence, 0.0)
        self.assertFalse(res.is_valid)

    def test_invalid_dimension_features(self):
        invalid_feat = np.zeros(25)
        res = self.predictor.predict_features(invalid_feat)
        self.assertEqual(res.label, "Unknown")
        self.assertFalse(res.is_valid)


if __name__ == "__main__":
    unittest.main()
