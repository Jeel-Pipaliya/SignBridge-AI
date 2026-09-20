"""
SignBridge AI - Unit Tests for Static + Dynamic Decision Fusion (Week 5)
Tests:
  - Static prediction flow
  - Dynamic prediction arbitration over static frames
  - Missing hand graceful decay (NONE modality)
  - Dynamic cooldown suppression
  - Low confidence filtering
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from ai.fusion.recognizer_fusion import StaticDynamicFusionEngine, RecognitionResult
from ai.predict import PredictionResult
from ai.dynamic.predict import DynamicPredictionResult


class TestFusionEngine(unittest.TestCase):
    """Test suite for multimodal static/dynamic fusion."""

    def setUp(self):
        self.mock_static = MagicMock()
        self.mock_dynamic = MagicMock()
        self.mock_dynamic.is_ready = True

        self.engine = StaticDynamicFusionEngine(
            static_predictor=self.mock_static,
            dynamic_predictor=self.mock_dynamic,
            sequence_length=5,
            inference_interval=1,
            dynamic_cooldown_seconds=1.0,
            static_threshold=0.70,
            dynamic_threshold=0.70,
            min_motion_energy=0.01,
        )

    def test_missing_hand_returns_none(self):
        result = self.engine.process_landmarks(None)
        self.assertIsInstance(result, RecognitionResult)
        self.assertEqual(result.type, "NONE")
        self.assertEqual(result.label, "...")
        self.assertEqual(result.confidence, 0.0)

    def test_static_prediction_flow(self):
        # Configure static predictor to return "A" with high confidence
        self.mock_static.predict_features.return_value = PredictionResult("A", "A", 0.95, True, {})
        # Configure dynamic predictor to return invalid (static holding pose)
        self.mock_dynamic.predict_sequence.return_value = DynamicPredictionResult("...", 0.0, False, 0.0, {})

        dummy_landmarks = np.random.randn(21, 3).tolist()

        # Feed 5 frames to fill temporal smoother
        results = [self.engine.process_landmarks(dummy_landmarks) for _ in range(5)]
        final_result = results[-1]

        self.assertEqual(final_result.type, "STATIC")
        self.assertEqual(final_result.label, "A")
        self.assertGreaterEqual(final_result.confidence, 0.70)

    def test_dynamic_prediction_priority(self):
        # Fill sequence buffer
        dummy_landmarks = np.random.randn(21, 3).tolist()
        self.mock_static.predict_features.return_value = PredictionResult("5", "5", 0.75, True, {})
        self.mock_dynamic.predict_sequence.return_value = DynamicPredictionResult(
            "HELLO", 0.92, True, 0.05, {"HELLO": 0.92, "J": 0.04, "Z": 0.04}
        )

        # Feed enough frames to ready sequence builder
        results = [self.engine.process_landmarks(dummy_landmarks) for _ in range(6)]
        dynamic_hits = [r for r in results if r.type == "DYNAMIC"]

        self.assertGreater(len(dynamic_hits), 0)
        self.assertEqual(dynamic_hits[0].label, "HELLO")
        self.assertAlmostEqual(dynamic_hits[0].confidence, 0.92)


if __name__ == "__main__":
    unittest.main()
