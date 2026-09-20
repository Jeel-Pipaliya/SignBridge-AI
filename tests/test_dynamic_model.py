"""
SignBridge AI - Unit Tests for Dynamic Bi-LSTM & Baseline Models (Week 5)
Tests:
  - BiLSTMClassifier topology and output dimensions
  - Softmax probability distributions (sum to 1.0)
  - Batch inference and single sequence inference
  - Checkpoint save and load fidelity
  - Metadata dictionary export
  - DynamicBaselineClassifier temporal feature extraction and fitting
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from ai.dynamic.model import BiLSTMClassifier, DynamicBaselineClassifier


class TestDynamicModel(unittest.TestCase):
    """Test suite for dynamic gesture models."""

    def test_bilstm_forward_pass_dimensions(self):
        batch_size = 4
        T, F = 30, 63
        num_classes = 3
        model = BiLSTMClassifier(
            sequence_length=T,
            feature_dim=F,
            hidden_size=128,
            num_classes=num_classes,
        )

        dummy_batch = torch.randn(batch_size, T, F)
        logits = model(dummy_batch)
        self.assertEqual(logits.shape, (batch_size, num_classes))

    def test_bilstm_predict_proba(self):
        T, F = 30, 63
        model = BiLSTMClassifier(sequence_length=T, feature_dim=F, num_classes=3)
        dummy_seq = np.random.randn(T, F).astype(np.float32)

        probs = model.predict_proba(dummy_seq)
        self.assertEqual(probs.shape, (1, 3))
        # Probabilities must sum to 1.0
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertTrue(np.all(probs >= 0.0))

    def test_bilstm_metadata(self):
        model = BiLSTMClassifier(sequence_length=30, feature_dim=63, hidden_size=128, num_classes=3)
        meta = model.get_metadata()
        self.assertEqual(meta["model_type"], "BiLSTM")
        self.assertEqual(meta["sequence_length"], 30)
        self.assertEqual(meta["feature_dim"], 63)
        self.assertEqual(meta["num_classes"], 3)
        self.assertGreater(meta["total_parameters"], 10000)

    def test_bilstm_checkpoint_save_and_load(self):
        model = BiLSTMClassifier(sequence_length=30, feature_dim=63, num_classes=3)
        dummy_input = np.random.randn(30, 63).astype(np.float32)
        orig_probs = model.predict_proba(dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_model.pt"
            model.save_checkpoint(ckpt_path, val_acc=0.95)
            self.assertTrue(ckpt_path.exists())

            loaded_model = BiLSTMClassifier.load_checkpoint(ckpt_path)
            loaded_probs = loaded_model.predict_proba(dummy_input)
            np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-5)

    def test_baseline_classifier_temporal_features(self):
        seq = np.random.randn(30, 63).astype(np.float32)
        feats = DynamicBaselineClassifier.extract_temporal_features(seq)
        # 5 statistics x 63 features = 315
        self.assertEqual(feats.shape, (315,))

    def test_baseline_classifier_fit_and_predict(self):
        X = np.random.randn(15, 30, 63).astype(np.float32)
        y = np.array([0, 1, 2] * 5)
        baseline = DynamicBaselineClassifier(model_type="random_forest", random_state=42)
        baseline.fit(X, y)

        preds = baseline.predict(X[:3])
        self.assertEqual(len(preds), 3)

        probs = baseline.predict_proba(X[:3])
        self.assertEqual(probs.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
