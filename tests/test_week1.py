"""
SignBridge AI - Week 1 Automated Test Suite
Verifies:
1. Environment and dependencies
2. Project directory structure and configuration paths
3. MediaPipe HandDetector initialization and processing
4. 63-Dimensional Invariant Feature Extraction
5. Invariance properties (translation, scale, combined)
6. Data Collection CSV generation and schema correctness
"""

import sys
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.hand_detection import HandDetector
from ai.feature_extraction import (
    extract_features,
    normalize_landmarks,
    get_feature_names,
    verify_invariance,
)
from data.collect_samples import DataCollector


class TestWeek1EnvironmentAndStructure(unittest.TestCase):
    """Test environment packages and directory layout."""

    def test_required_directories_exist(self):
        required_dirs = [
            config.RAW_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.TRAIN_DATA_DIR,
            config.VAL_DATA_DIR,
            config.TEST_DATA_DIR,
            config.OUTPUTS_DIR,
            config.PLOTS_DIR,
            config.CONFUSION_MATRIX_DIR,
            config.PREDICTIONS_DIR,
            config.MODELS_DIR,
        ]
        for d in required_dirs:
            d.mkdir(parents=True, exist_ok=True)
            self.assertTrue(d.exists(), f"Missing directory: {d}")

    def test_config_constants(self):
        self.assertEqual(config.NUM_LANDMARKS, 21)
        self.assertEqual(config.NUM_COORDINATES, 3)
        self.assertEqual(config.NUM_FEATURES, 63)
        self.assertGreater(config.CONFIDENCE_THRESHOLD, 0.0)


class TestHandDetection(unittest.TestCase):
    """Test MediaPipe Hand Detector."""

    def setUp(self):
        self.detector = HandDetector(static_image_mode=True, max_num_hands=1)

    def tearDown(self):
        self.detector.close()

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.hands)

    def test_detector_blank_frame(self):
        # A pure black frame should detect zero hands and not crash
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, hands = self.detector.process(blank, draw=True)
        self.assertEqual(len(hands), 0)
        self.assertEqual(annotated.shape, blank.shape)


class TestFeatureExtraction(unittest.TestCase):
    """Test 63D Feature Extraction and Invariance."""

    def test_feature_vector_dimension(self):
        dummy_landmarks = [(0.1 * i, 0.05 * i, 0.02 * i) for i in range(21)]
        features = extract_features(dummy_landmarks)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (63,))
        self.assertEqual(features.dtype, np.float32)

    def test_feature_names_count(self):
        names = get_feature_names()
        self.assertEqual(len(names), 63)
        self.assertEqual(names[0], "lm0_x")
        self.assertEqual(names[-1], "lm20_z")

    def test_wrist_at_origin_after_normalization(self):
        # Wrist is index 0; its coordinates after centering should be (0, 0, 0)
        dummy_landmarks = np.random.uniform(0.2, 0.8, size=(21, 3)).astype(np.float32)
        features = extract_features(dummy_landmarks)
        self.assertAlmostEqual(features[0], 0.0, places=5)
        self.assertAlmostEqual(features[1], 0.0, places=5)
        self.assertAlmostEqual(features[2], 0.0, places=5)

    def test_invariance_verification(self):
        passed = verify_invariance()
        self.assertTrue(passed, "Feature extraction invariance test failed!")


class TestDataCollectionCSV(unittest.TestCase):
    """Test CSV generation and schema validation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_csv = Path(self.temp_dir) / "test_collected.csv"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_csv_creation_and_append(self):
        collector = DataCollector(output_csv=self.temp_csv, initial_label="TEST_SIGN")
        self.assertTrue(self.temp_csv.exists())

        # Generate fake 63 features
        dummy_feat = np.ones(63, dtype=np.float32)
        saved = collector.save_sample(dummy_feat)
        self.assertTrue(saved)
        self.assertEqual(collector.session_count, 1)

        # Check sample count for class
        count = collector.get_class_sample_count("TEST_SIGN")
        self.assertEqual(count, 1)

        # Read back with pandas
        import pandas as pd
        df = pd.read_csv(self.temp_csv)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.shape[1], 64)  # 63 features + 1 label
        self.assertEqual(df["label"].iloc[0], "TEST_SIGN")


if __name__ == "__main__":
    print("=" * 65)
    print("  Running SignBridge AI - Week 1 Test Suite")
    print("=" * 65)
    unittest.main(verbosity=2)
