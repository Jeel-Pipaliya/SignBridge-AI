"""
SignBridge AI - Automated Dataset Validation & Health Check (Week 4 Phase 1)
Validates image files, dimensions, integrity, labels, and processed landmarks.
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, Set
import cv2
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.utils.logger import setup_logger

logger = setup_logger("DatasetValidator")


def validate_dataset(max_images_per_class: int = 100) -> Dict[str, Any]:
    """
    Scans raw image directories and processed dataset files for corruption,
    empty files, invalid extensions, extreme dimensions, duplicates, and missing labels.
    """
    raw_dir = config.RAW_DATA_DIR
    processed_file = config.LANDMARKS_CSV_PATH

    total_images_checked = 0
    valid_count = 0
    corrupt_count = 0
    missing_labels_count = 0
    invalid_files_count = 0
    duplicate_count = 0
    small_images_count = 0

    seen_hashes: Set[str] = set()

    if not raw_dir.exists():
        logger.error(f"Raw dataset directory does not exist at {raw_dir}")
        status = "ERROR"
    else:
        classes = [d for d in os.listdir(raw_dir) if (raw_dir / d).is_dir()]
        for cls in classes:
            class_dir = raw_dir / cls
            files = os.listdir(class_dir)
            # Scan files (up to max_images_per_class to ensure fast execution while maintaining deep audit)
            for f in files[:max_images_per_class]:
                total_images_checked += 1
                f_path = class_dir / f

                # Check extension
                if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    invalid_files_count += 1
                    continue

                # Check empty file
                if f_path.stat().st_size == 0:
                    corrupt_count += 1
                    continue

                # Compute MD5 for duplicate detection
                try:
                    with open(f_path, "rb") as fp:
                        file_hash = hashlib.md5(fp.read()).hexdigest()
                    if file_hash in seen_hashes:
                        duplicate_count += 1
                    else:
                        seen_hashes.add(file_hash)
                except Exception:
                    corrupt_count += 1
                    continue

                # Image decode test
                try:
                    img = cv2.imread(str(f_path))
                    if img is None or img.size == 0:
                        corrupt_count += 1
                        continue

                    h, w = img.shape[:2]
                    if h < 32 or w < 32:
                        small_images_count += 1

                    valid_count += 1
                except Exception:
                    corrupt_count += 1

    # Validate Processed Landmarks CSV
    if processed_file.exists():
        try:
            df = pd.read_csv(processed_file)
            if "label" not in df.columns or df["label"].isnull().any():
                missing_labels_count += int(df["label"].isnull().sum())
            # Check for NaN features
            feat_cols = [c for c in df.columns if c.startswith("lm")]
            if df[feat_cols].isnull().any().any():
                corrupt_count += int(df[feat_cols].isnull().any(axis=1).sum())
        except Exception as exc:
            logger.error(f"Failed to parse processed landmarks CSV: {exc}")
            corrupt_count += 1

    status = "OK" if corrupt_count == 0 and missing_labels_count == 0 and invalid_files_count == 0 else "WARNING"

    print("\nDataset Validation")
    print("------------------")
    print(f"Images checked:  {total_images_checked:,}")
    print(f"Valid:           {valid_count:,}")
    print(f"Corrupt:         {corrupt_count}")
    print(f"Missing labels:  {missing_labels_count}")
    print(f"Invalid files:   {invalid_files_count}")
    print(f"Duplicates:      {duplicate_count}")
    print(f"Small (<32px):   {small_images_count}")
    print(f"\nStatus: {status}\n")

    return {
        "images_checked": total_images_checked,
        "valid": valid_count,
        "corrupt": corrupt_count,
        "missing_labels": missing_labels_count,
        "invalid_files": invalid_files_count,
        "duplicates": duplicate_count,
        "status": status,
    }


if __name__ == "__main__":
    validate_dataset()
