import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    PreprocessingResult,
    batch_preprocess_dataset,
)

__all__ = [
    "ImagePreprocessor",
    "PreprocessingResult",
    "batch_preprocess_dataset",
]

if __name__ == "__main__":
    total, succ, out_csv = batch_preprocess_dataset()
    print(f"Batch preprocessing completed: {succ}/{total} samples extracted.")
