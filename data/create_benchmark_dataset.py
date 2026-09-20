"""
SignBridge AI - Landmark Benchmark Dataset Generator (Week 2 Day 6)
Constructs anatomically grounded 21-landmark configurations for all 12 ISL classes.
Applies natural biometric variability (hand size, joint angles, camera perspective jitter, noise)
and processes every sample through the invariant 63D feature normalization engine.

Generates:
  data/processed/processed_landmarks.csv
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.feature_extraction import normalize_landmarks, get_feature_names
from data.ingest_dataset import PROTOTYPE_CLASSES, ensure_directories


def build_base_hand_landmarks(finger_states: Dict[str, str]) -> np.ndarray:
    """
    Constructs 21 3D coordinates (x, y, z) based on anatomical finger extensions:
      'extended': finger straight upright
      'folded':   finger curled inward into palm
      'curved':   finger arched into 'C' curve
      'outward':  finger spread laterally outward
    """
    lms = np.zeros((21, 3), dtype=np.float32)

    # Wrist: Landmark 0
    lms[0] = [0.50, 0.80, 0.0]

    # Thumb: Landmarks 1-4
    thumb_state = finger_states.get("thumb", "folded")
    lms[1] = [0.46, 0.74, -0.01]  # CMC
    lms[2] = [0.42, 0.67, -0.02]  # MCP
    if thumb_state == "outward":
        lms[3] = [0.35, 0.63, -0.03]  # IP
        lms[4] = [0.29, 0.60, -0.04]  # TIP
    elif thumb_state == "extended":
        lms[3] = [0.38, 0.58, -0.03]
        lms[4] = [0.34, 0.51, -0.04]
    elif thumb_state == "curved":
        lms[3] = [0.40, 0.64, -0.04]
        lms[4] = [0.43, 0.61, -0.05]
    else:  # folded across palm
        lms[3] = [0.46, 0.65, -0.04]
        lms[4] = [0.50, 0.64, -0.04]

    # Index: Landmarks 5-8
    idx_state = finger_states.get("index", "folded")
    lms[5] = [0.46, 0.59, -0.01]  # MCP
    if idx_state == "spread":
        lms[6] = [0.42, 0.49, -0.02]  # PIP angled left
        lms[7] = [0.38, 0.40, -0.03]  # DIP angled left
        lms[8] = [0.34, 0.30, -0.04]  # TIP angled left in V-shape
    elif idx_state == "extended":
        lms[6] = [0.45, 0.49, -0.02]  # PIP
        lms[7] = [0.44, 0.41, -0.03]  # DIP
        lms[8] = [0.43, 0.33, -0.04]  # TIP
    elif idx_state == "curved":
        lms[6] = [0.45, 0.51, -0.04]
        lms[7] = [0.43, 0.46, -0.06]
        lms[8] = [0.41, 0.49, -0.07]
    else:  # folded
        lms[6] = [0.46, 0.62, -0.03]
        lms[7] = [0.47, 0.66, -0.04]
        lms[8] = [0.48, 0.69, -0.04]

    # Middle: Landmarks 9-12
    mid_state = finger_states.get("middle", "folded")
    lms[9] = [0.50, 0.58, -0.01]  # MCP
    if mid_state == "spread":
        lms[10] = [0.53, 0.47, -0.02]  # PIP angled right
        lms[11] = [0.57, 0.39, -0.03]  # DIP angled right
        lms[12] = [0.61, 0.30, -0.04]  # TIP angled right in V-shape
    elif mid_state == "extended":
        lms[10] = [0.50, 0.47, -0.02]  # PIP
        lms[11] = [0.50, 0.39, -0.03]  # DIP
        lms[12] = [0.50, 0.30, -0.04]  # TIP
    elif mid_state == "curved":
        lms[10] = [0.50, 0.50, -0.04]
        lms[11] = [0.49, 0.45, -0.06]
        lms[12] = [0.47, 0.48, -0.07]
    else:  # folded
        lms[10] = [0.50, 0.62, -0.03]
        lms[11] = [0.50, 0.66, -0.04]
        lms[12] = [0.50, 0.69, -0.04]

    # Ring: Landmarks 13-16
    rng_state = finger_states.get("ring", "folded")
    lms[13] = [0.54, 0.60, -0.01]  # MCP
    if rng_state == "spread":
        lms[14] = [0.57, 0.50, -0.02]  # PIP angled right
        lms[15] = [0.60, 0.42, -0.03]  # DIP angled right
        lms[16] = [0.64, 0.33, -0.04]  # TIP angled right
    elif rng_state == "extended":
        lms[14] = [0.55, 0.50, -0.02]  # PIP
        lms[15] = [0.55, 0.42, -0.03]  # DIP
        lms[16] = [0.55, 0.35, -0.04]  # TIP
    elif rng_state == "curved":
        lms[14] = [0.54, 0.52, -0.04]
        lms[15] = [0.53, 0.47, -0.06]
        lms[16] = [0.51, 0.50, -0.07]
    else:  # folded
        lms[14] = [0.53, 0.63, -0.03]
        lms[15] = [0.53, 0.66, -0.04]
        lms[16] = [0.52, 0.69, -0.04]

    # Pinky: Landmarks 17-20
    pky_state = finger_states.get("pinky", "folded")
    lms[17] = [0.58, 0.63, -0.01]  # MCP
    if pky_state == "outward":
        lms[18] = [0.62, 0.55, -0.02]  # PIP
        lms[19] = [0.65, 0.48, -0.03]  # DIP
        lms[20] = [0.68, 0.41, -0.04]  # TIP
    elif pky_state == "extended":
        lms[18] = [0.59, 0.54, -0.02]
        lms[19] = [0.60, 0.47, -0.03]
        lms[20] = [0.60, 0.40, -0.04]
    elif pky_state == "curved":
        lms[18] = [0.58, 0.54, -0.04]
        lms[19] = [0.57, 0.49, -0.06]
        lms[20] = [0.55, 0.52, -0.07]
    else:  # folded
        lms[18] = [0.57, 0.65, -0.03]
        lms[19] = [0.56, 0.68, -0.04]
        lms[20] = [0.55, 0.70, -0.04]

    return lms


CLASS_FINGER_CONFIGS: Dict[str, Dict[str, str]] = {
    # Numbers
    "1":     {"thumb": "folded", "index": "extended", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "2":     {"thumb": "folded", "index": "extended", "middle": "extended", "ring": "folded", "pinky": "folded"},
    "3":     {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "folded", "pinky": "folded"},
    "4":     {"thumb": "folded", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"},
    "5":     {"thumb": "outward", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"},
    # Alphabets (ISL)
    "A":     {"thumb": "extended", "index": "folded", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "B":     {"thumb": "folded", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"},
    "C":     {"thumb": "curved", "index": "curved", "middle": "curved", "ring": "curved", "pinky": "curved"},
    "D":     {"thumb": "curved", "index": "extended", "middle": "curved", "ring": "curved", "pinky": "curved"},
    "E":     {"thumb": "folded", "index": "curved", "middle": "curved", "ring": "curved", "pinky": "curved"},
    "F":     {"thumb": "curved", "index": "curved", "middle": "extended", "ring": "extended", "pinky": "extended"},
    "G":     {"thumb": "outward", "index": "extended", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "H":     {"thumb": "folded", "index": "extended", "middle": "extended", "ring": "folded", "pinky": "folded"},
    "I":     {"thumb": "folded", "index": "folded", "middle": "folded", "ring": "folded", "pinky": "extended"},
    "J":     {"thumb": "folded", "index": "folded", "middle": "folded", "ring": "folded", "pinky": "curved"},
    "K":     {"thumb": "outward", "index": "extended", "middle": "curved", "ring": "folded", "pinky": "folded"},
    "L":     {"thumb": "outward", "index": "extended", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "M":     {"thumb": "folded", "index": "folded", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "N":     {"thumb": "folded", "index": "folded", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "O":     {"thumb": "curved", "index": "curved", "middle": "curved", "ring": "curved", "pinky": "curved"},
    "P":     {"thumb": "outward", "index": "extended", "middle": "curved", "ring": "folded", "pinky": "folded"},
    "Q":     {"thumb": "curved", "index": "curved", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "R":     {"thumb": "folded", "index": "extended", "middle": "extended", "ring": "folded", "pinky": "folded"},
    "S":     {"thumb": "folded", "index": "folded", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "T":     {"thumb": "extended", "index": "curved", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "U":     {"thumb": "folded", "index": "extended", "middle": "extended", "ring": "folded", "pinky": "folded"},
    "V":     {"thumb": "folded", "index": "spread", "middle": "spread", "ring": "folded", "pinky": "folded"},
    "W":     {"thumb": "folded", "index": "spread", "middle": "extended", "ring": "spread", "pinky": "folded"},
    "X":     {"thumb": "folded", "index": "curved", "middle": "folded", "ring": "folded", "pinky": "folded"},
    "Y":     {"thumb": "outward", "index": "folded", "middle": "folded", "ring": "folded", "pinky": "outward"},
    "Z":     {"thumb": "folded", "index": "extended", "middle": "folded", "ring": "folded", "pinky": "folded"},
    # Prototype Words
    "HELLO": {"thumb": "outward", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"},
}


def create_benchmark_features(
    samples_per_class: int = 50,
    output_csv: Path = config.LANDMARKS_CSV_PATH,
) -> pd.DataFrame:
    """
    Generates realistic 63D normalized feature dataset for all 12 classes with natural jitter.
    """
    ensure_directories()
    headers = get_feature_names() + ["label", "handedness", "confidence", "image_path"]

    rows = []
    np.random.seed(1337)

    for cls in PROTOTYPE_CLASSES:
        states = CLASS_FINGER_CONFIGS.get(cls, {})
        base_lms = build_base_hand_landmarks(states)

        for i in range(samples_per_class):
            # Apply subtle joint jitter
            jitter = np.random.normal(0, 0.008, base_lms.shape).astype(np.float32)
            varied_lms = base_lms + jitter

            # Random scale variation
            scale_factor = np.random.uniform(0.85, 1.25)
            wrist = varied_lms[0]
            scaled_lms = wrist + (varied_lms - wrist) * scale_factor

            # Random translation in frame
            shift = np.random.uniform(-0.15, 0.15, size=(1, 3)).astype(np.float32)
            frame_lms = scaled_lms + shift

            # Extract 63D normalized features
            features = normalize_landmarks(frame_lms)

            confidence = round(float(np.random.uniform(0.92, 0.99)), 4)
            handedness = "Right" if i % 4 != 0 else "Left"
            img_ref = f"bench_{cls}_{i+1:03d}.jpg"

            row = list(features) + [cls, handedness, confidence, img_ref]
            rows.append(row)

    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(output_csv, index=False)
    print(f"[Success] Generated benchmark feature dataset with {len(df)} samples across {len(PROTOTYPE_CLASSES)} classes.")
    print(f"Saved to: {output_csv}")
    return df


if __name__ == "__main__":
    create_benchmark_features(samples_per_class=50)
