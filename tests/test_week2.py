"""
SignBridge AI - Week 2 Automated Test Suite
Verifies:
1. Dataset research and vocabulary documentation
2. Class mapping integrity (12 prototype classes)
3. EDA plot generation (distribution & sample grid)
4. ImagePreprocessor pipeline & error handling
5. Processed feature CSV schema (63 numerical features + label)
6. Stratified train/val/test partitions (70 / 15 / 15 ratio)
7. LabelEncoder serialization and bidirectional mapping
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
from data.ingest_dataset import PROTOTYPE_CLASSES, load_class_mapping
from ai.preprocessing import ImagePreprocessor
from ai.dataset_loader import DatasetLoader, TRAIN_CSV, VAL_CSV, TEST_CSV


class TestWeek2DocumentationAndVocabulary(unittest.TestCase):
    """Test research documentation and vocabulary definitions."""

    def test_research_docs_exist(self):
        research_doc = config.DOCS_DIR / "isl_dataset_research.md"
        vocab_doc = config.DOCS_DIR / "prototype_vocabulary.md"
        self.assertTrue(research_doc.exists(), "Missing isl_dataset_research.md")
        self.assertTrue(vocab_doc.exists(), "Missing prototype_vocabulary.md")

    def test_prototype_classes_count(self):
        self.assertEqual(len(PROTOTYPE_CLASSES), 12)
        mapping = load_class_mapping()
        self.assertGreaterEqual(len(mapping), 12)
        for cls in PROTOTYPE_CLASSES:
            self.assertIn(cls, mapping)


class TestWeek2EDAVisualizations(unittest.TestCase):
    """Test EDA plot generation."""

    def test_plots_exist(self):
        dist_plot = config.PLOTS_DIR / "class_distribution.png"
        grid_plot = config.PLOTS_DIR / "sample_grid.png"
        self.assertTrue(dist_plot.exists(), "Missing class_distribution.png")
        self.assertTrue(grid_plot.exists(), "Missing sample_grid.png")
        self.assertGreater(dist_plot.stat().st_size, 0)
        self.assertGreater(grid_plot.stat().st_size, 0)


class TestWeek2PreprocessingAndFeatures(unittest.TestCase):
    """Test batch preprocessing and feature extraction dataset."""

    def test_preprocessor_blank_image(self):
        with ImagePreprocessor() as prep:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            res = prep.process_single_image(blank, label="TEST")
            self.assertFalse(res.success)
            self.assertIsNone(res.features)
            self.assertIn("No hand landmarks", res.error_reason)

    def test_processed_landmarks_csv_schema(self):
        self.assertTrue(config.LANDMARKS_CSV_PATH.exists())
        df = pd.read_csv(config.LANDMARKS_CSV_PATH)

        self.assertGreaterEqual(len(df), 120, "Expected at least 120 samples in processed dataset")
        self.assertIn("label", df.columns)

        # Count feature columns
        feat_cols = [c for c in df.columns if c.startswith("lm")]
        self.assertEqual(len(feat_cols), config.NUM_FEATURES)

        # Check for NaN / infinite values
        features_matrix = df[feat_cols].values
        self.assertFalse(np.isnan(features_matrix).any(), "Found NaN values in feature matrix!")
        self.assertFalse(np.isinf(features_matrix).any(), "Found Infinite values in feature matrix!")


class TestWeek2StratifiedSplits(unittest.TestCase):
    """Test stratified split partitions and label encoder."""

    def test_split_files_exist(self):
        self.assertTrue(TRAIN_CSV.exists(), "Missing train.csv")
        self.assertTrue(VAL_CSV.exists(), "Missing val.csv")
        self.assertTrue(TEST_CSV.exists(), "Missing test.csv")

    def test_split_ratios_and_classes(self):
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VAL_CSV)
        df_test = pd.read_csv(TEST_CSV)

        total = len(df_train) + len(df_val) + len(df_test)
        self.assertGreater(total, 0)

        # Check ratios approximate 70% / 15% / 15%
        train_ratio = len(df_train) / total
        val_ratio = len(df_val) / total
        test_ratio = len(df_test) / total

        self.assertAlmostEqual(train_ratio, 0.70, delta=0.03)
        self.assertAlmostEqual(val_ratio, 0.15, delta=0.03)
        self.assertAlmostEqual(test_ratio, 0.15, delta=0.03)

        # Check classes are present in train, val, and test
        num_classes = len(df_train["label"].unique())
        self.assertGreaterEqual(num_classes, 12)
        self.assertEqual(len(df_val["label"].unique()), num_classes)
        self.assertEqual(len(df_test["label"].unique()), num_classes)

    def test_label_encoder_bidirectional(self):
        self.assertTrue(config.LABEL_ENCODER_PATH.exists())
        le = joblib.load(config.LABEL_ENCODER_PATH)

        self.assertGreaterEqual(len(le.classes_), 12)
        encoded = le.transform(["HELLO", "1", "A"])
        decoded = le.inverse_transform(encoded)
        self.assertEqual(list(decoded), ["HELLO", "1", "A"])


if __name__ == "__main__":
    print("=" * 65)
    print("  Running SignBridge AI - Week 2 Test Suite")
    print("=" * 65)
    unittest.main(verbosity=2)
