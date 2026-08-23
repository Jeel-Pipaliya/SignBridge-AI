"""
DAY 6 — SignBridge AI
Sequence Builder: assembles per-frame landmark vectors into fixed-length sequences.

A "sequence" is the fundamental training unit for the LSTM/GRU model in Week 3.

    Frame 1  → 225 features
    Frame 2  → 225 features
    ...
    Frame 30 → 225 features
    ─────────────────────────
    Sequence shape → (30, 225)

Usage:
    builder = SequenceBuilder(sequence_length=30)
    builder.add_frame(landmark_vector)      # call once per webcam frame
    if builder.is_ready():
        seq = builder.get_sequence()        # np.ndarray (30, 225)
        builder.reset()                     # or use sliding window
"""

import numpy as np
from typing import Optional


class SequenceBuilder:
    """
    Accumulates per-frame landmark vectors and emits fixed-length sequences.

    Attributes:
        sequence_length: Number of frames per sequence (default 30).
        frames:          Internal buffer.
    """

    def __init__(self, sequence_length: int = 30) -> None:
        self.sequence_length = sequence_length
        self.frames: list[np.ndarray] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add_frame(self, landmark_vector: np.ndarray) -> None:
        """Append one frame's landmark vector to the buffer."""
        self.frames.append(np.asarray(landmark_vector, dtype=np.float32))

    def is_ready(self) -> bool:
        """True when enough frames have been collected."""
        return len(self.frames) >= self.sequence_length

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Return the last `sequence_length` frames as a (T, F) array,
        where T = sequence_length and F = number of features.

        Returns None if not enough frames have been added yet.
        """
        if not self.is_ready():
            return None
        seq = np.array(self.frames[-self.sequence_length :], dtype=np.float32)
        return seq

    def reset(self) -> None:
        """Clear the frame buffer (use after saving a complete sequence)."""
        self.frames = []

    def progress(self) -> tuple[int, int]:
        """Return (frames_collected, sequence_length) for UI display."""
        return min(len(self.frames), self.sequence_length), self.sequence_length

    # ── Convenience class method ──────────────────────────────────────────────

    @classmethod
    def from_frames(
        cls,
        frames: list[np.ndarray],
        sequence_length: int = 30,
    ) -> Optional[np.ndarray]:
        """
        One-shot helper: given a list of frames return a sequence array,
        or None if there are not enough frames.

        Example:
            seq = SequenceBuilder.from_frames(all_frames, sequence_length=30)
        """
        if len(frames) < sequence_length:
            return None
        return np.array(frames[-sequence_length:], dtype=np.float32)


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SequenceBuilder test")

    builder = SequenceBuilder(sequence_length=30)

    # Simulate 40 frames × 225 features
    for _ in range(40):
        fake_frame = np.random.rand(225).astype(np.float32)
        builder.add_frame(fake_frame)

    if builder.is_ready():
        seq = builder.get_sequence()
        print(f"  Sequence shape : {seq.shape}")   # expected (30, 225)
        print(f"  dtype          : {seq.dtype}")
        print("  PASS")
    else:
        print("  FAIL — not enough frames")
