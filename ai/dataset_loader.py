"""
SignBridge AI - Dataset Loader & Stratified Train/Val/Test Split Module (Week 2 Day 7)
Loads processed landmark CSVs, encodes categorical labels, applies stratified partitioning,
and serializes partitioned dataframes and the fitted LabelEncoder.

Partitions:
  - Training:   70%
  - Validation: 15%
  - Testing:    15%
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Output split paths
TRAIN_CSV = config.PROCESSED_DATA_DIR / "train.csv"
VAL_CSV = config.PROCESSED_DATA_DIR / "val.csv"
TEST_CSV = config.PROCESSED_DATA_DIR / "test.csv"


class DatasetLoader:
    """
    Handles feature data loading, stratified train/validation/test splitting,
    and label encoding for SignBridge AI.
    """

    def __init__(
        self,
        csv_path: Path = config.LANDMARKS_CSV_PATH,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.classes_: np.ndarray = np.array([])

    def load_raw_features(self) -> pd.DataFrame:
        """Load feature CSV into a pandas DataFrame and validate schema."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}. "
                f"Please run data collection or preprocessing first."
            )

        df = pd.read_csv(self.csv_path)
        if "label" not in df.columns:
            raise ValueError(f"Required column 'label' missing from {self.csv_path}")

        # Extract 63 feature column names
        self.feature_columns = [c for c in df.columns if c.startswith("lm")]
        if len(self.feature_columns) != config.NUM_FEATURES:
            # Fallback to taking first 63 columns
            self.feature_columns = list(df.columns[: config.NUM_FEATURES])

        logger.info(f"Loaded {len(df)} samples with {len(self.feature_columns)} features from {self.csv_path.name}")
        return df

    def prepare_stratified_splits(
        self,
        save_splits: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads dataset, applies label encoding, performs 70/15/15 stratified split,
        saves split CSV files and serialized label encoder.

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        df = self.load_raw_features()

        X = df[self.feature_columns].values.astype(np.float32)
        raw_labels = df["label"].astype(str).values

        # Fit label encoder
        y = self.label_encoder.fit_transform(raw_labels)
        self.classes_ = self.label_encoder.classes_

        # Save LabelEncoder artifact
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoder, config.LABEL_ENCODER_PATH)
        logger.info(f"LabelEncoder saved to {config.LABEL_ENCODER_PATH}")

        # 1. First split: Train+Val (85%) vs Test (15%)
        X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
            X,
            y,
            np.arange(len(y)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # 2. Second split: Train (70% total) vs Val (15% total)
        # val_ratio relative to train_val is val_size / (1.0 - test_size)
        val_relative_ratio = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_train_val,
            y_train_val,
            idx_train_val,
            test_size=val_relative_ratio,
            random_state=self.random_state,
            stratify=y_train_val,
        )

        logger.info(
            f"Stratified Split: "
            f"Train={len(X_train)} ({len(X_train)/len(X):.1%}), "
            f"Val={len(X_val)} ({len(X_val)/len(X):.1%}), "
            f"Test={len(X_test)} ({len(X_test)/len(X):.1%})"
        )

        if save_splits:
            df.iloc[idx_train].to_csv(TRAIN_CSV, index=False)
            df.iloc[idx_val].to_csv(VAL_CSV, index=False)
            df.iloc[idx_test].to_csv(TEST_CSV, index=False)
            logger.info(f"Split CSVs saved: train.csv, val.csv, test.csv in {config.PROCESSED_DATA_DIR}")

        return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataset_splits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Convenience function to load or create stratified splits and return them.
    """
    loader = DatasetLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.prepare_stratified_splits(save_splits=True)
    return X_train, y_train, X_val, y_val, X_test, y_test, loader.label_encoder


if __name__ == "__main__":
    print("=" * 65)
    print("  SignBridge AI - Dataset Loader & Stratified Splitting (Week 2 Day 7)")
    print("=" * 65)
    X_train, y_train, X_val, y_val, X_test, y_test, le = load_dataset_splits()
    print("\n[Splits Summary]")
    print(f"  • Features Shape (X_train): {X_train.shape}")
    print(f"  • Target Shape   (y_train): {y_train.shape}")
    print(f"  • Validation Set (X_val):   {X_val.shape}")
    print(f"  • Test Set       (X_test):  {X_test.shape}")
    print(f"  • Classes ({len(le.classes_)}): {list(le.classes_)}")
