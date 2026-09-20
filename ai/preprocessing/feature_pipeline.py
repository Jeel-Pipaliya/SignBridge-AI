"""
SignBridge AI - End-to-End Feature Pipeline Module (Week 4 Phase 2)
Encapsulates landmark extraction, invariant normalization, validation,
and caching of feature matrices under data/landmarks/.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.preprocessing.landmark_extractor import extract_landmarks_from_hand, extract_landmarks
from ai.preprocessing.normalization import normalize_landmarks
from ai.utils.logger import setup_logger

logger = setup_logger("FeaturePipeline")


class FeaturePipeline:
    """
    Standardized Feature Pipeline:
    Transforms detected hand landmarks into normalized, model-ready feature vectors.
    """

    def __init__(self, input_dim: int = 63, scale_method: str = "middle_mcp"):
        self.input_dim = input_dim
        self.scale_method = scale_method

    def process_landmarks(self, landmarks: Any) -> Optional[np.ndarray]:
        """
        Extracts raw landmarks and normalizes them into an invariant 1D vector.
        Returns: np.ndarray of shape (input_dim,) or None if hand missing.
        """
        if landmarks is None:
            return None

        raw = extract_landmarks(landmarks)
        if not np.any(raw):
            return None

        try:
            normalized = normalize_landmarks(raw, scale_method=self.scale_method)
            return normalized
        except Exception as exc:
            logger.warning(f"Normalization failed: {exc}")
            return None

    def prepare_model_input(self, features: Union[list, np.ndarray]) -> np.ndarray:
        """
        Formats 1D feature array into 2D batch tensor of shape (1, input_dim).
        """
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, got {arr.shape[1]}"
            )
        return arr

    def cache_processed_splits(
        self,
        cache_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path, Path]:
        """
        Loads CSV splits (train, val, test) and caches feature matrices
        and label arrays into fast binary .npy format under data/landmarks/.
        """
        if cache_dir is None:
            cache_dir = config.DATA_DIR / "landmarks"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        splits = ["train", "val", "test"]
        saved_paths = []

        for split in splits:
            csv_path = config.PROCESSED_DATA_DIR / f"{split}.csv"
            if not csv_path.exists():
                logger.warning(f"Cannot cache split '{split}': {csv_path} does not exist.")
                continue

            df = pd.read_csv(csv_path)
            feat_cols = [c for c in df.columns if c.startswith("lm")]
            X = df[feat_cols].values.astype(np.float32)
            y = df["label"].astype(str).values

            split_dir = cache_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            x_out = split_dir / "features.npy"
            y_out = split_dir / "labels.npy"

            np.save(str(x_out), X)
            np.save(str(y_out), y)
            logger.info(f"Cached {split} split to {split_dir}: X={X.shape}, y={y.shape}")
            saved_paths.append(split_dir)

        return tuple(saved_paths)


def get_feature_pipeline() -> FeaturePipeline:
    """Factory helper returning standard FeaturePipeline instance."""
    return FeaturePipeline(input_dim=config.NUM_FEATURES)


if __name__ == "__main__":
    pipeline = get_feature_pipeline()
    pipeline.cache_processed_splits()
