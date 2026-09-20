"""
SignBridge AI - Dynamic Gesture Dataset Loader & Synthetic Generator (Week 5)
Handles:
  1. Loading and parsing recorded NPZ dynamic landmark sequences from disk.
  2. Synthetic sequence generation for pipeline testing and verification (HELLO, J, Z).
  3. Stratified Train / Validation / Test dataset partitioning.
  4. PyTorch Dataset and DataLoader integration.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.dynamic.preprocessing import (
    preprocess_dynamic_sequence,
    augment_sequence,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_FEATURE_DIM,
)
from ai.preprocessing.normalization import normalize_landmarks
from ai.utils.logger import setup_logger

logger = setup_logger("DynamicDataset")

DYNAMIC_CLASSES = ["HELLO", "J", "Z"]
LABEL_TO_IDX = {cls: idx for idx, cls in enumerate(DYNAMIC_CLASSES)}
IDX_TO_LABEL = {idx: cls for idx, cls in enumerate(DYNAMIC_CLASSES)}


class DynamicSequenceDataset:
    """
    In-memory container for dynamic gesture sequences.
    Holds X of shape (N, T, F) and y of shape (N,).
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        sequence_ids: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
    ):
        self.sequences = np.asarray(sequences, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes = classes if classes is not None else DYNAMIC_CLASSES
        self.sequence_ids = (
            sequence_ids
            if sequence_ids is not None
            else [f"seq_{i:04d}" for i in range(len(self.sequences))]
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        return self.sequences[idx], int(self.labels[idx]), self.sequence_ids[idx]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.sequences.shape


def generate_synthetic_gesture(
    gesture_class: str,
    target_length: int = DEFAULT_SEQUENCE_LENGTH,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generates a realistic kinematic trajectory for dynamic gestures:
      - HELLO: Hand palm open, oscillating lateral wave (sinusoidal sweep in X-axis).
      - J: Index finger / pinky extended, vertical downward stroke then curved hook.
      - Z: Index finger tracing three segments (top bar -> diagonal slash -> bottom bar).
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    sequence = np.zeros((target_length, 21, 3), dtype=np.float32)

    # Base open hand structure relative to wrist at (0, 0, 0)
    # Point 0: Wrist at (0, 0, 0)
    # Finger MCPs: points 1, 5, 9, 13, 17
    # Middle MCP (point 9) at (0.0, 0.40, 0.0) sets the reference anatomical distance
    base_hand = np.zeros((21, 3), dtype=np.float32)
    # Thumb (1..4)
    base_hand[1:5] = [[-0.12, 0.15, 0.0], [-0.20, 0.25, 0.0], [-0.25, 0.35, 0.0], [-0.28, 0.42, 0.0]]
    # Index (5..8)
    base_hand[5:9] = [[-0.08, 0.38, 0.0], [-0.09, 0.52, 0.0], [-0.10, 0.64, 0.0], [-0.10, 0.74, 0.0]]
    # Middle (9..12)
    base_hand[9:13] = [[0.00, 0.40, 0.0], [0.00, 0.56, 0.0], [0.00, 0.69, 0.0], [0.00, 0.80, 0.0]]
    # Ring (13..16)
    base_hand[13:17] = [[0.08, 0.38, 0.0], [0.09, 0.52, 0.0], [0.10, 0.64, 0.0], [0.10, 0.74, 0.0]]
    # Pinky (17..20)
    base_hand[17:21] = [[0.15, 0.32, 0.0], [0.18, 0.44, 0.0], [0.20, 0.54, 0.0], [0.22, 0.62, 0.0]]

    # Curled finger coordinates (folded into palm)
    curled_index = np.array([[-0.08, 0.38, 0.0], [-0.07, 0.30, 0.1], [-0.06, 0.22, 0.05], [-0.05, 0.20, 0.0]], dtype=np.float32)
    curled_middle = np.array([[0.00, 0.40, 0.0], [0.00, 0.31, 0.1], [0.00, 0.23, 0.05], [0.00, 0.20, 0.0]], dtype=np.float32)
    curled_ring = np.array([[0.08, 0.38, 0.0], [0.07, 0.30, 0.1], [0.06, 0.22, 0.05], [0.05, 0.20, 0.0]], dtype=np.float32)
    curled_pinky = np.array([[0.15, 0.32, 0.0], [0.14, 0.25, 0.1], [0.12, 0.18, 0.05], [0.10, 0.16, 0.0]], dtype=np.float32)

    if gesture_class == "HELLO":
        # Dynamic Waving Motion: All fingers open, oscillating angular rotation theta around wrist
        freq = float(rng.uniform(2.0, 3.2))
        max_angle = float(rng.uniform(0.30, 0.45))  # in radians (~20-25 degrees)
        for idx in range(target_length):
            theta = max_angle * np.sin(2 * np.pi * freq * t[idx])
            rot_mat = np.array([
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta),  np.cos(theta), 0.0],
                [0.0,            0.0,           1.0],
            ], dtype=np.float32)
            sequence[idx] = base_hand @ rot_mat.T

    elif gesture_class == "J":
        # J Stroke: Pinky (or index) extended tracing a downward curve and hook
        # Fold other fingers into fist
        fist = base_hand.copy()
        fist[5:9] = curled_index
        fist[9:13] = curled_middle
        fist[13:17] = curled_ring

        for idx in range(target_length):
            frame = fist.copy()
            progress = t[idx]
            if progress < 0.60:
                # Downward movement
                dy = -0.30 * (progress / 0.60)
                dx = 0.0
            else:
                # Hook curving left
                hook_t = (progress - 0.60) / 0.40
                dx = -0.25 * np.sin(hook_t * np.pi * 0.75)
                dy = -0.30 + 0.15 * (1.0 - np.cos(hook_t * np.pi * 0.75))

            # Move pinky tip and joints according to stroke
            frame[17:21, 0] += dx
            frame[17:21, 1] += dy
            sequence[idx] = frame

    elif gesture_class == "Z":
        # Z Stroke: Pointing gesture (index finger extended, middle/ring/pinky curled)
        pointing = base_hand.copy()
        pointing[9:13] = curled_middle
        pointing[13:17] = curled_ring
        pointing[17:21] = curled_pinky

        for idx in range(target_length):
            frame = pointing.copy()
            progress = t[idx]
            if progress < 0.33:
                # Stroke 1: Left to right
                s1 = progress / 0.33
                dx = -0.15 + s1 * 0.30
                dy = 0.10
            elif progress < 0.66:
                # Stroke 2: Diagonal down-left
                s2 = (progress - 0.33) / 0.33
                dx = 0.15 - s2 * 0.30
                dy = 0.10 - s2 * 0.20
            else:
                # Stroke 3: Bottom horizontal right
                s3 = (progress - 0.66) / 0.34
                dx = -0.15 + s3 * 0.30
                dy = -0.10

            # Displace index finger along Z stroke trajectory
            frame[5:9, 0] += dx
            frame[5:9, 1] += dy
            sequence[idx] = frame
    else:
        raise ValueError(f"Unknown dynamic gesture class: '{gesture_class}'")

    # Apply Week 4 invariant normalization per frame
    norm_seq = np.zeros((target_length, feature_dim), dtype=np.float32)
    for i in range(target_length):
        norm_seq[i] = normalize_landmarks(sequence[i].flatten())

    return norm_seq


def create_synthetic_dataset(
    samples_per_class: int = 120,
    target_length: int = DEFAULT_SEQUENCE_LENGTH,
    augment: bool = True,
    random_seed: int = 42,
) -> DynamicSequenceDataset:
    """
    Creates a calibrated synthetic dynamic dataset for HELLO, J, and Z.
    Used for automated test suites, CI validation, and baseline development.
    """
    rng = np.random.default_rng(random_seed)
    all_sequences = []
    all_labels = []
    all_ids = []

    for cls_name in DYNAMIC_CLASSES:
        cls_idx = LABEL_TO_IDX[cls_name]
        for i in range(samples_per_class):
            base_seq = generate_synthetic_gesture(
                gesture_class=cls_name,
                target_length=target_length,
                rng=rng,
            )
            if augment:
                seq = augment_sequence(base_seq, rng=rng)
            else:
                seq = base_seq
            all_sequences.append(seq)
            all_labels.append(cls_idx)
            all_ids.append(f"{cls_name}_{i:04d}")

    # Shuffle dataset
    indices = np.arange(len(all_sequences))
    rng.shuffle(indices)

    seq_arr = np.array(all_sequences, dtype=np.float32)[indices]
    lbl_arr = np.array(all_labels, dtype=np.int64)[indices]
    ids_arr = [all_ids[i] for i in indices]

    logger.info(f"Generated synthetic dynamic dataset: {len(seq_arr)} sequences across {len(DYNAMIC_CLASSES)} classes.")
    return DynamicSequenceDataset(seq_arr, lbl_arr, ids_arr, classes=DYNAMIC_CLASSES)


def load_dynamic_dataset(
    data_dir: Optional[Union[str, Path]] = None,
    target_length: int = DEFAULT_SEQUENCE_LENGTH,
    fallback_synthetic: bool = True,
) -> DynamicSequenceDataset:
    """
    Loads saved NPZ sequences from disk.
    If no sequences exist and fallback_synthetic=True, generates a synthetic dataset.
    """
    if data_dir is None:
        data_dir = config.DYNAMIC_SEQUENCES_DIR

    data_dir = Path(data_dir)
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    sequence_ids: List[str] = []

    if data_dir.exists():
        for npz_path in data_dir.glob("*.npz"):
            try:
                data = np.load(npz_path, allow_pickle=True)
                raw_seq = data["sequence"]
                label_str = str(data["label"])
                if label_str not in LABEL_TO_IDX:
                    continue
                preprocessed = preprocess_dynamic_sequence(raw_seq, target_length=target_length)
                sequences.append(preprocessed)
                labels.append(LABEL_TO_IDX[label_str])
                sequence_ids.append(npz_path.stem)
            except Exception as err:
                logger.warning(f"Could not load sequence file {npz_path}: {err}")

    if len(sequences) == 0:
        if fallback_synthetic:
            logger.info("No recorded dynamic sequences found on disk. Generating synthetic dataset.")
            return create_synthetic_dataset(samples_per_class=100, target_length=target_length)
        else:
            raise FileNotFoundError(f"No dynamic gesture sequences found in {data_dir}.")

    seq_arr = np.array(sequences, dtype=np.float32)
    lbl_arr = np.array(labels, dtype=np.int64)
    logger.info(f"Loaded {len(seq_arr)} dynamic sequences from {data_dir}.")
    return DynamicSequenceDataset(seq_arr, lbl_arr, sequence_ids, classes=DYNAMIC_CLASSES)


def split_dataset(
    dataset: DynamicSequenceDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[DynamicSequenceDataset, DynamicSequenceDataset, DynamicSequenceDataset]:
    """
    Stratified split into Train, Validation, and Test subsets.
    """
    rng = np.random.default_rng(random_seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls_idx in range(len(dataset.classes)):
        cls_mask = np.where(dataset.labels == cls_idx)[0]
        rng.shuffle(cls_mask)
        n = len(cls_mask)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_idx.extend(cls_mask[:n_train])
        val_idx.extend(cls_mask[n_train : n_train + n_val])
        test_idx.extend(cls_mask[n_train + n_val :])

    train_ds = DynamicSequenceDataset(
        dataset.sequences[train_idx],
        dataset.labels[train_idx],
        [dataset.sequence_ids[i] for i in train_idx],
        classes=dataset.classes,
    )
    val_ds = DynamicSequenceDataset(
        dataset.sequences[val_idx],
        dataset.labels[val_idx],
        [dataset.sequence_ids[i] for i in val_idx],
        classes=dataset.classes,
    )
    test_ds = DynamicSequenceDataset(
        dataset.sequences[test_idx],
        dataset.labels[test_idx],
        [dataset.sequence_ids[i] for i in test_idx],
        classes=dataset.classes,
    )

    logger.info(f"Dataset split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    return train_ds, val_ds, test_ds
