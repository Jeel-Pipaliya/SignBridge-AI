"""
SignBridge AI - Unit Tests for ISL Classifier Model Architecture (Week 4)
"""

import unittest
import numpy as np

from ai.models.isl_classifier import ISLClassifier, build_isl_model


class TestISLModel(unittest.TestCase):
    """Test model construction, dynamic class dimensions, and inference."""

    def test_mlp_model_construction(self):
        model = build_isl_model(input_dim=63, num_classes=10, model_type="mlp")
        self.assertEqual(model.input_dim, 63)
        self.assertEqual(model.num_classes, 10)
        self.assertEqual(model.model_type, "mlp")

    def test_dynamic_class_fitting(self):
        # 5 classes test
        num_classes = 5
        X = np.random.uniform(-1.0, 1.0, size=(50, 63)).astype(np.float32)
        y = np.random.randint(0, num_classes, size=(50,))

        model = ISLClassifier(input_dim=63, num_classes=num_classes, max_iter=50)
        model.fit(X, y)

        preds = model.predict(X[:5])
        self.assertEqual(len(preds), 5)

        probs = model.predict_proba(X[:5])
        self.assertEqual(probs.shape, (5, num_classes))
        # Probabilities sum to 1
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-5)

    def test_metadata_export(self):
        model = ISLClassifier(input_dim=63, num_classes=32)
        meta = model.get_metadata()
        self.assertIn("model_type", meta)
        self.assertIn("input_dim", meta)
        self.assertIn("num_classes", meta)
        self.assertEqual(meta["input_dim"], 63)
        self.assertEqual(meta["num_classes"], 32)

    def test_invalid_input_dim_raises_error(self):
        model = ISLClassifier(input_dim=63, num_classes=10)
        invalid_X = np.random.randn(10, 50)
        with self.assertRaises(ValueError):
            model.fit(invalid_X, np.zeros(10))


if __name__ == "__main__":
    unittest.main()
