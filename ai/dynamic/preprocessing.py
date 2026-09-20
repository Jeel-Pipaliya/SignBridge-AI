"""
SignBridge AI - Dynamic Sequence Preprocessing & Augmentation Pipeline (Week 5)
Handles:
  1. Landmark feature normalization (wrist-centered, anatomical scale invariant).
  2. Temporal resampling & interpolation to fixed sequence length (T=30).
  3. Padding / truncation for variable-length gesture sequences.
  4. Sequence-level data augmentation (jitter, scaling, frame drop, time-warp).
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.preprocessing.normalization import normalize_landmarks, NUM_LANDMARKS, NUM_COORDS

DEFAULT_SEQUENCE_LENGTH: int = 30
DEFAULT_FEATURE_DIM: int = NUM_LANDMARKS * NUM_COORDS  # 63


def resample_sequence(
    sequence: np.ndarray,
    target_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Resample or linearly interpolate a variable-length sequence of shape (N, F)
    to exactly (target_length, F).

    Args:
        sequence: Array of shape (N, F), where N >= 1 and F is feature dimension.
        target_length: Desired number of temporal frames (T).

    Returns:
        Resampled array of shape (target_length, F), dtype float32.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)

    n_frames, n_features = seq.shape

    if n_frames == target_length:
        return seq.copy()

    if n_frames == 1:
        # Replicate single frame across the entire target length
        return np.repeat(seq, target_length, axis=0)

    # Linear interpolation along the temporal dimension
    orig_indices = np.linspace(0.0, 1.0, num=n_frames, endpoint=True)
    target_indices = np.linspace(0.0, 1.0, num=target_length, endpoint=True)

    resampled = np.zeros((target_length, n_features), dtype=np.float32)
    for feat_idx in range(n_features):
        resampled[:, feat_idx] = np.interp(target_indices, orig_indices, seq[:, feat_idx])

    return resampled


def pad_or_truncate_sequence(
    sequence: np.ndarray,
    target_length: int = DEFAULT_SEQUENCE_LENGTH,
    pad_mode: str = "edge",
) -> np.ndarray:
    """
    Adjusts sequence to target_length by either truncating (if N > target_length)
    or padding (if N < target_length).

    Args:
        sequence: Array of shape (N, F).
        target_length: Desired length.
        pad_mode: 'edge' (repeat boundary frames) or 'zero' (zero padding).

    Returns:
        Array of shape (target_length, F).
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)

    n_frames, n_features = seq.shape

    if n_frames == target_length:
        return seq.copy()

    if n_frames > target_length:
        # Uniform subsampling or centered crop
        step = n_frames / target_length
        indices = [int(i * step) for i in range(target_length)]
        return seq[indices].copy()

    # Need padding
    pad_width = target_length - n_frames
    if pad_mode == "zero":
        padding = np.zeros((pad_width, n_features), dtype=np.float32)
        return np.vstack([seq, padding])
    elif pad_mode == "edge":
        last_frame = seq[-1:, :]
        padding = np.repeat(last_frame, pad_width, axis=0)
        return np.vstack([seq, padding])
    else:
        raise ValueError(f"Unsupported pad_mode: '{pad_mode}'")


def preprocess_dynamic_sequence(
    sequence: Union[List[np.ndarray], np.ndarray],
    target_length: int = DEFAULT_SEQUENCE_LENGTH,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    normalize_per_frame: bool = False,
) -> np.ndarray:
    """
    Unified preprocessing entrypoint:
      - Validates shape and dimension.
      - Optionally applies Week 4 63D landmark normalization per frame if raw.
      - Resamples smoothly to target_length (T=30).

    Args:
        sequence: Iterable or ndarray of frame vectors (N, F).
        target_length: Desired sequence length.
        feature_dim: Expected feature dimension (63).
        normalize_per_frame: If True, each frame is passed to normalize_landmarks().

    Returns:
        np.ndarray of shape (target_length, feature_dim), dtype float32.
    """
    if len(sequence) == 0:
        raise ValueError("Cannot preprocess an empty sequence.")

    frames = []
    for frame in sequence:
        f = np.asarray(frame, dtype=np.float32).flatten()
        if normalize_per_frame:
            f = normalize_landmarks(f)
        if f.size != feature_dim:
            raise ValueError(f"Frame feature size mismatch: expected {feature_dim}, got {f.size}")
        frames.append(f)

    seq_arr = np.stack(frames, axis=0)
    return resample_sequence(seq_arr, target_length=target_length)


def augment_sequence(
    sequence: np.ndarray,
    noise_std: float = 0.015,
    scale_range: Tuple[float, float] = (0.92, 1.08),
    shift_range: Tuple[float, float] = (-0.03, 0.03),
    temporal_stretch_range: Tuple[float, float] = (0.85, 1.15),
    drop_rate: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Biologically sound data augmentation for landmark coordinate sequences:
      - Gaussian coordinate noise (small camera sensor jitter).
      - Random scaling (signer distance variance).
      - Random spatial translation shift (signer positioning variance).
      - Temporal stretching / compression (gestural speed variance).
      - Frame dropout with linear interpolation recovery.

    Args:
        sequence: Array of shape (T, 63).
        noise_std: Standard deviation of additive Gaussian noise.
        scale_range: Min/max scale multiplier.
        shift_range: Min/max translation shift.
        temporal_stretch_range: Speed variation ratio.
        drop_rate: Probability of randomly dropping a non-boundary frame.
        rng: Optional NumPy random Generator.

    Returns:
        Augmented array of shape (T, 63), dtype float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    T, F = sequence.shape
    aug = sequence.copy()

    # 1. Temporal Stretch / Compression
    stretch_factor = float(rng.uniform(*temporal_stretch_range))
    new_len = max(5, int(round(T * stretch_factor)))
    stretched = resample_sequence(aug, target_length=new_len)
    aug = resample_sequence(stretched, target_length=T)

    # 2. Random Frame Drop
    if drop_rate > 0.0:
        keep_mask = rng.random(T) > drop_rate
        keep_mask[0] = True   # Keep first frame
        keep_mask[-1] = True  # Keep last frame
        if np.sum(keep_mask) >= 3:
            aug = resample_sequence(aug[keep_mask], target_length=T)

    # 3. Additive Gaussian Noise
    if noise_std > 0.0:
        noise = rng.normal(0.0, noise_std, size=(T, F)).astype(np.float32)
        aug += noise

    # 4. Global Scale Variation
    scale = float(rng.uniform(*scale_range))
    aug *= scale

    # 5. Spatial Translation Shift
    shift = float(rng.uniform(*shift_range))
    aug += shift

    return aug.astype(np.float32)


def compute_sequence_motion_energy(sequence: np.ndarray) -> float:
    """
    Computes the average inter-frame motion delta across all landmarks.
    Used by the Static/Dynamic fusion layer to differentiate static holding from
    active dynamic gestures.

    Args:
        sequence: Array of shape (T, 63).

    Returns:
        Mean Euclidean displacement per frame across landmarks.
    """
    if len(sequence) < 2:
        return 0.0
    deltas = np.diff(sequence, axis=0)  # Shape (T-1, 63)
    # Average absolute displacement across all coordinates and frames
    energy = float(np.mean(np.linalg.norm(deltas.reshape(-1, NUM_LANDMARKS, NUM_COORDS), axis=-1)))
    return energy
