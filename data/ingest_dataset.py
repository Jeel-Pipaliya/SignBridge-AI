"""
SignBridge AI - Dataset Ingestion & Organization Module (Week 2 Day 3)
Manages dataset directory layout, verifies raw image integrity, validates class structures,
and creates official class-to-index mappings.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard 12-class prototype vocabulary
PROTOTYPE_CLASSES = ["1", "2", "3", "4", "5", "A", "B", "C", "L", "V", "Y", "HELLO"]
MAPPING_FILE = config.DATA_DIR / "class_mapping.json"


def ensure_directories() -> None:
    """Create all required data subdirectories if they do not exist."""
    directories = [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.TRAIN_DATA_DIR,
        config.VAL_DATA_DIR,
        config.TEST_DATA_DIR,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("All data directories verified.")


def generate_class_mapping(classes: List[str]) -> Dict[str, int]:
    """Create and save deterministic class-to-index dictionary."""
    sorted_classes = sorted(classes)
    mapping = {label: idx for idx, label in enumerate(sorted_classes)}
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved class mapping for {len(mapping)} classes to {MAPPING_FILE}")
    return mapping


def load_class_mapping() -> Dict[str, int]:
    """Load existing class mapping, or generate default prototype mapping."""
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return generate_class_mapping(PROTOTYPE_CLASSES)


def scan_raw_dataset(raw_dir: Path = config.RAW_DATA_DIR) -> Dict[str, List[Path]]:
    """
    Scan data/raw/ directory for subdirectories representing sign classes.
    Returns:
        dict: {class_name: [list of valid image Paths]}
    """
    class_images: Dict[str, List[Path]] = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not raw_dir.exists():
        return class_images

    # Iterate over subdirectories
    for class_folder in sorted(raw_dir.iterdir()):
        if class_folder.is_dir() and not class_folder.name.startswith("."):
            images = []
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    images.append(img_path)
            if images:
                class_images[class_folder.name] = images

    return class_images


def verify_image_integrity(image_path: Path) -> bool:
    """Check if an image file is uncorrupted and can be decoded by OpenCV."""
    try:
        img = cv2.imread(str(image_path))
        if img is None or img.size == 0:
            return False
        return True
    except Exception:
        return False


def setup_prototype_folders() -> None:
    """Initialize empty class directories for the 12 prototype classes in data/raw/."""
    ensure_directories()
    for cls in PROTOTYPE_CLASSES:
        cls_dir = config.RAW_DATA_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
    generate_class_mapping(PROTOTYPE_CLASSES)
    print(f"Created prototype directories for {len(PROTOTYPE_CLASSES)} classes in {config.RAW_DATA_DIR}")


if __name__ == "__main__":
    print("=" * 65)
    print("  SignBridge AI - Dataset Organization & Ingestion (Week 2 Day 3)")
    print("=" * 65)
    setup_prototype_folders()
    scanned = scan_raw_dataset()
    print(f"Scanned {len(scanned)} classes with images in {config.RAW_DATA_DIR}")
    for cls, imgs in scanned.items():
        print(f"  - Class '{cls}': {len(imgs)} images")
