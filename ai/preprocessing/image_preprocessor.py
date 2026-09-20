"""
SignBridge AI - Image Preprocessing & Batch Extraction Module (Week 2 Day 5 & 6)
Processes sign language images through:
1. Image loading & validation (OpenCV)
2. Color conversion (BGR -> RGB)
3. MediaPipe Hand Landmark Detection
4. Translation- and Scale-Invariant Normalization (63-dimensional vector)
5. Graceful handling and logging of frames where no hand is detected
6. Batch processing of image directory into processed_landmarks.csv
"""

import sys
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
import cv2
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.hand_detection import HandDetector
from ai.feature_extraction import extract_features, get_feature_names

logger = logging.getLogger(__name__)


class PreprocessingResult(NamedTuple):
    """Encapsulates the output of preprocessing a single image."""
    success: bool
    features: Optional[np.ndarray]  # Shape (63,) if success
    label: str
    image_path: str
    handedness: str
    confidence: float
    error_reason: Optional[str] = None


class ImagePreprocessor:
    """
    Robust image preprocessor for ISL datasets.
    Reuses HandDetector and feature normalization engines.
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
    ):
        self.target_size = target_size
        self.detector = HandDetector(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
        )

    def process_single_image(
        self,
        image_input: np.ndarray,
        label: str = "UNKNOWN",
        image_path: str = "",
    ) -> PreprocessingResult:
        """
        Process a single image array through the pipeline:
        OpenCV Decode -> Resize -> MediaPipe Hands -> Invariant 63D Normalizer
        """
        if image_input is None or image_input.size == 0:
            return PreprocessingResult(
                success=False,
                features=None,
                label=label,
                image_path=image_path,
                handedness="Unknown",
                confidence=0.0,
                error_reason="Empty or invalid image array",
            )

        # Standardize size if requested
        if self.target_size is not None:
            h, w = image_input.shape[:2]
            if (w, h) != self.target_size:
                image_input = cv2.resize(image_input, self.target_size, interpolation=cv2.INTER_AREA)

        # Run hand detection
        _, hands = self.detector.process(image_input, draw=False)

        if not hands:
            return PreprocessingResult(
                success=False,
                features=None,
                label=label,
                image_path=image_path,
                handedness="None",
                confidence=0.0,
                error_reason="No hand landmarks detected by MediaPipe",
            )

        primary_hand = hands[0]
        features = extract_features(primary_hand.landmark_list)

        if features is None or len(features) != config.NUM_FEATURES:
            return PreprocessingResult(
                success=False,
                features=None,
                label=label,
                image_path=image_path,
                handedness=primary_hand.handedness,
                confidence=primary_hand.confidence,
                error_reason="Feature extraction failed or produced invalid dimension",
            )

        return PreprocessingResult(
            success=True,
            features=features,
            label=label,
            image_path=image_path,
            handedness=primary_hand.handedness,
            confidence=primary_hand.confidence,
            error_reason=None,
        )

    def process_image_file(self, file_path: Path, label: str) -> PreprocessingResult:
        """Load image from disk and process it."""
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                return PreprocessingResult(
                    success=False,
                    features=None,
                    label=label,
                    image_path=str(file_path),
                    handedness="Unknown",
                    confidence=0.0,
                    error_reason="Could not decode image file with OpenCV",
                )
            return self.process_single_image(img, label=label, image_path=str(file_path))
        except Exception as exc:
            return PreprocessingResult(
                success=False,
                features=None,
                label=label,
                image_path=str(file_path),
                handedness="Unknown",
                confidence=0.0,
                error_reason=f"Exception during processing: {str(exc)}",
            )

    def close(self) -> None:
        """Close MediaPipe detector resources."""
        self.detector.close()

    def __enter__(self) -> "ImagePreprocessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def batch_preprocess_dataset(
    raw_dir: Path = config.RAW_DATA_DIR,
    output_csv: Path = config.LANDMARKS_CSV_PATH,
    min_confidence: float = 0.50,
) -> Tuple[int, int, Path]:
    """
    Scans data/raw/<class>/*.jpg, runs feature extraction on each image,
    and compiles valid feature vectors into output_csv.

    Returns:
        (total_processed: int, total_successful: int, output_csv: Path)
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    headers = get_feature_names() + ["label", "handedness", "confidence", "image_path"]

    total_processed = 0
    total_successful = 0
    failed_reasons: Dict[str, int] = {}

    rows = []

    print(f"Scanning raw images in: {raw_dir}...")
    with ImagePreprocessor(min_detection_confidence=min_confidence) as preprocessor:
        for class_dir in sorted(raw_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue

            cls_name = class_dir.name
            img_files = [p for p in class_dir.iterdir() if p.suffix.lower() in valid_extensions]

            if not img_files:
                continue

            print(f"  Processing Class '{cls_name}': {len(img_files)} images...")
            for img_p in img_files:
                total_processed += 1
                result = preprocessor.process_image_file(img_p, label=cls_name)

                if result.success and result.features is not None:
                    row = list(result.features) + [
                        result.label,
                        result.handedness,
                        round(result.confidence, 4),
                        str(img_p.name),
                    ]
                    rows.append(row)
                    total_successful += 1
                else:
                    reason = result.error_reason or "Unknown"
                    failed_reasons[reason] = failed_reasons.get(reason, 0) + 1

    # Save to CSV
    if rows:
        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(output_csv, index=False)
        print(f"\n[Success] Saved {len(df)} feature records to: {output_csv}")
    else:
        print("\n[Warning] No valid features were extracted from raw images.")

    if failed_reasons:
        print("\nDropped/Failed Samples Breakdown:")
        for r, cnt in failed_reasons.items():
            print(f"  - {r}: {cnt} samples")

    return total_processed, total_successful, output_csv


if __name__ == "__main__":
    total, succ, out_csv = batch_preprocess_dataset()
    print(f"Batch preprocessing completed: {succ}/{total} samples extracted.")
