"""
SignBridge AI - Unit Tests for Utilities & Metrics (Week 4)
"""

import unittest
import numpy as np

from ai.utils.config import load_config, get_resolved_path
from ai.utils.logger import setup_logger
from ai.utils.metrics import compute_metrics, find_top_misclassifications


class TestUtilities(unittest.TestCase):
    """Test configuration, logger, and metrics utilities."""

    def test_load_config(self):
        cfg = load_config()
        self.assertIn("model", cfg)
        self.assertIn("training", cfg)
        self.assertIn("inference", cfg)
        self.assertIn("paths", cfg)
        self.assertEqual(cfg["model"]["input_dim"], 63)

    def test_resolved_path(self):
        path = get_resolved_path("configs/training.yaml")
        self.assertTrue(path.exists())

    def test_logger_creation(self):
        logger = setup_logger("TestLogger")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "TestLogger")

    def test_compute_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        classes = ["A", "B", "C"]

        metrics = compute_metrics(y_true, y_pred, classes)
        # 5 out of 6 correct = 83.33%
        self.assertAlmostEqual(metrics["accuracy"], 5 / 6, places=4)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_top_misclassifications(self):
        cm = np.array([
            [5, 1, 0],
            [2, 4, 0],
            [0, 0, 6],
        ])
        classes = ["A", "B", "C"]
        errors = find_top_misclassifications(cm, classes, top_k=2)
        # B -> A occurred 2 times
        self.assertEqual(errors[0], ("B", "A", 2))
        # A -> B occurred 1 time
        self.assertEqual(errors[1], ("A", "B", 1))


if __name__ == "__main__":
    unittest.main()
