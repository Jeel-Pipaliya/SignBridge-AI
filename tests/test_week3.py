"""
SignBridge AI - Week 3 Automated Test Suite
Verifies:
1. Model artifacts existence & serialization
2. SignPredictor inference & confidence bounds
3. TemporalSmoother sliding-window majority voting & flicker elimination
4. TextAccumulator token commit, debouncing, backspace, and clear operations
5. UI HUD overlay rendering on dummy frames
6. Model evaluation artifacts (confusion matrix & per-class plots)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.predict import SignPredictor, PredictionResult
from realtime.recognizer import TemporalSmoother, TextAccumulator
from realtime.realtime_demo import draw_ui_overlay


class TestWeek3ModelArtifacts(unittest.TestCase):
    """Test model checkpoints and serialized artifacts."""

    def test_model_files_exist(self):
        self.assertTrue(config.MODEL_PATH.exists(), f"Missing {config.MODEL_PATH}")
        self.assertTrue(config.LABEL_ENCODER_PATH.exists(), f"Missing {config.LABEL_ENCODER_PATH}")

    def test_model_loading(self):
        model = joblib.load(config.MODEL_PATH)
        encoder = joblib.load(config.LABEL_ENCODER_PATH)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(len(encoder.classes_), 12)


class TestWeek3SignPredictor(unittest.TestCase):
    """Test SignPredictor inference logic."""

    def setUp(self):
        self.predictor = SignPredictor()

    def test_predict_features_output_schema(self):
        dummy_feat = np.random.uniform(-1.0, 1.0, size=(63,)).astype(np.float32)
        res = self.predictor.predict_features(dummy_feat)

        self.assertIsInstance(res, PredictionResult)
        self.assertIsInstance(res.label, str)
        self.assertIsInstance(res.raw_label, str)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)
        self.assertIn(res.raw_label, self.predictor.classes)

    def test_predict_invalid_features(self):
        invalid_feat = np.zeros(10)
        res = self.predictor.predict_features(invalid_feat)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.label, "Unknown")


class TestWeek3TemporalSmoother(unittest.TestCase):
    """Test majority voting temporal smoother."""

    def test_flicker_rejection(self):
        smoother = TemporalSmoother(window_size=5, min_agreement_ratio=0.6)

        # Feed 1 HELLO, 1 NOISE, then 3 HELLO
        p_hello = PredictionResult("HELLO", "HELLO", 0.95, True, {})
        p_noise = PredictionResult("NOISE", "NOISE", 0.80, True, {})

        s1, _ = smoother.update(p_hello)
        s2, _ = smoother.update(p_noise)
        s3, _ = smoother.update(p_hello)
        s4, _ = smoother.update(p_hello)
        s5, c5 = smoother.update(p_hello)

        # After 4/5 HELLO, majority vote should strongly be HELLO
        self.assertEqual(s5, "HELLO")
        self.assertGreater(c5, 0.9)


class TestWeek3TextAccumulator(unittest.TestCase):
    """Test text formation, debouncing, backspace, and clear."""

    def test_accumulation_and_controls(self):
        acc = TextAccumulator(confirmation_frames=3, debounce_seconds=0.1)

        # Stable input for 3 frames
        acc.update("HELLO")
        acc.update("HELLO")
        committed = acc.update("HELLO")
        self.assertEqual(committed, "HELLO")
        self.assertEqual(acc.get_text(), "HELLO")

        # Insert space
        acc.add_space()
        self.assertEqual(acc.get_text(), "HELLO")

        # Stable input for another sign
        acc.update("A")
        acc.update("A")
        acc.update("A")
        self.assertIn("A", acc.get_text())

        # Backspace
        acc.backspace()
        self.assertEqual(acc.get_text(), "HELLO")

        # Clear
        acc.clear()
        self.assertEqual(acc.get_text(), "...")


class TestWeek3UIOverlay(unittest.TestCase):
    """Test visual HUD rendering."""

    def test_overlay_renders_on_frame(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = draw_ui_overlay(
            frame=dummy_frame,
            smoothed_sign="HELLO",
            confidence=0.94,
            fps=29.5,
            accumulated_text="HELLO WORLD",
            hand_count=1,
        )
        self.assertEqual(rendered.shape, (480, 640, 3))


class TestWeek3EvaluationArtifacts(unittest.TestCase):
    """Test presence of evaluation outputs."""

    def test_evaluation_plots_exist(self):
        cm_path = config.CONFUSION_MATRIX_DIR / "confusion_matrix.png"
        f1_path = config.PLOTS_DIR / "per_class_f1.png"
        self.assertTrue(cm_path.exists(), "Missing confusion_matrix.png")
        self.assertTrue(f1_path.exists(), "Missing per_class_f1.png")


if __name__ == "__main__":
    print("=" * 65)
    print("  Running SignBridge AI - Week 3 Test Suite")
    print("=" * 65)
    unittest.main(verbosity=2)
