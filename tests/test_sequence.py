"""
DAY 6 -- SignBridge AI
Tests for the SequenceBuilder module.

Run:
    python tests/test_sequence.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.preprocessing.sequence_builder import SequenceBuilder
from ai.preprocessing.landmark_extractor import TOTAL_FEATURES


SEQUENCE_LENGTH = 30


def _pass(name: str) -> None:
    print(f"  [OK]  {name}")


def _fail(name: str, reason: str) -> None:
    print(f"  [!!]  {name} -- {reason}")
    sys.exit(1)


# ---- Tests -------------------------------------------------------------------

def test_not_ready_before_enough_frames():
    builder = SequenceBuilder(SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH - 1):
        builder.add_frame(np.zeros(TOTAL_FEATURES, dtype=np.float32))
    assert not builder.is_ready(), "Should NOT be ready with fewer frames"
    _pass("is_ready() = False when < sequence_length frames")


def test_ready_at_sequence_length():
    builder = SequenceBuilder(SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        builder.add_frame(np.zeros(TOTAL_FEATURES, dtype=np.float32))
    assert builder.is_ready(), "Should be ready at sequence_length"
    _pass("is_ready() = True at sequence_length frames")


def test_sequence_shape():
    builder = SequenceBuilder(SEQUENCE_LENGTH)
    for _ in range(40):
        builder.add_frame(np.random.rand(TOTAL_FEATURES).astype(np.float32))
    seq = builder.get_sequence()
    assert seq is not None, "get_sequence() returned None"
    assert seq.shape == (SEQUENCE_LENGTH, TOTAL_FEATURES), f"Wrong shape: {seq.shape}"
    _pass(f"Sequence shape == ({SEQUENCE_LENGTH}, {TOTAL_FEATURES})")


def test_get_sequence_returns_none_when_not_ready():
    builder = SequenceBuilder(SEQUENCE_LENGTH)
    result = builder.get_sequence()
    assert result is None, "Should return None before enough frames"
    _pass("get_sequence() returns None before enough frames")


def test_reset():
    builder = SequenceBuilder(SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        builder.add_frame(np.zeros(TOTAL_FEATURES, dtype=np.float32))
    assert builder.is_ready()
    builder.reset()
    assert not builder.is_ready(), "Should NOT be ready after reset"
    _pass("reset() clears the buffer")


def test_sliding_window():
    """get_sequence() should always use the LAST sequence_length frames."""
    builder = SequenceBuilder(SEQUENCE_LENGTH)
    rng = np.random.default_rng(0)
    frames = [rng.random(TOTAL_FEATURES).astype(np.float32) for _ in range(40)]
    for f in frames:
        builder.add_frame(f)
    seq = builder.get_sequence()
    expected = np.array(frames[-SEQUENCE_LENGTH:], dtype=np.float32)
    assert np.allclose(seq, expected), "Sliding window returned wrong frames"
    _pass("Sliding window returns last sequence_length frames")


def test_dtype():
    builder = SequenceBuilder(SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        builder.add_frame(np.ones(TOTAL_FEATURES, dtype=np.float64))
    seq = builder.get_sequence()
    assert seq.dtype == np.float32, f"Wrong dtype: {seq.dtype}"
    _pass("Sequence dtype is float32")


def test_from_frames_helper():
    rng = np.random.default_rng(1)
    frames = [rng.random(TOTAL_FEATURES).astype(np.float32) for _ in range(35)]
    seq = SequenceBuilder.from_frames(frames, sequence_length=SEQUENCE_LENGTH)
    assert seq is not None and seq.shape == (SEQUENCE_LENGTH, TOTAL_FEATURES)
    _pass(f"from_frames() helper returns correct shape ({SEQUENCE_LENGTH}, {TOTAL_FEATURES})")


def test_from_frames_insufficient():
    frames = [np.zeros(TOTAL_FEATURES, dtype=np.float32) for _ in range(10)]
    seq = SequenceBuilder.from_frames(frames, sequence_length=SEQUENCE_LENGTH)
    assert seq is None, "Should return None for insufficient frames"
    _pass("from_frames() returns None when frames < sequence_length")


def test_progress():
    builder = SequenceBuilder(SEQUENCE_LENGTH)
    for i in range(15):
        builder.add_frame(np.zeros(TOTAL_FEATURES, dtype=np.float32))
    done, total = builder.progress()
    assert done == 15 and total == SEQUENCE_LENGTH, f"Progress wrong: {done}/{total}"
    _pass(f"progress() reports 15/{SEQUENCE_LENGTH}")


# ---- Runner ------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  SignBridge AI -- SequenceBuilder Tests (Day 6)")
    print("=" * 55)

    test_not_ready_before_enough_frames()
    test_ready_at_sequence_length()
    test_sequence_shape()
    test_get_sequence_returns_none_when_not_ready()
    test_reset()
    test_sliding_window()
    test_dtype()
    test_from_frames_helper()
    test_from_frames_insufficient()
    test_progress()

    print()
    print("  All tests passed!")
    print("=" * 55)


if __name__ == "__main__":
    main()
