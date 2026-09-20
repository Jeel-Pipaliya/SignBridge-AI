"""
SignBridge AI - Temporal Smoothing & Text Formation Engine (Week 3 Day 5 & 6)
Implements:
1. TemporalSmoother: Sliding-window majority vote to eliminate single-frame prediction flicker.
2. TextAccumulator: Stable sign aggregation buffer with debounce logic, backspace, and clear controls.
"""

import sys
import time
from collections import deque, Counter
from pathlib import Path
from typing import Deque, List, Optional, Tuple

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.predict import PredictionResult


class TemporalSmoother:
    """
    Sliding-Window Majority Voting Engine.
    Filters raw frame-by-frame predictions to deliver a stable, flicker-free sign output.
    """

    def __init__(
        self,
        window_size: int = config.SMOOTHING_WINDOW,
        min_agreement_ratio: float = 0.60,
    ):
        self.window_size = window_size
        self.min_agreement_ratio = min_agreement_ratio

        # Window holds tuples of (label, confidence)
        self.window: Deque[Tuple[str, float]] = deque(maxlen=window_size)

    def update(self, prediction: PredictionResult) -> Tuple[str, float]:
        """
        Ingest a new raw prediction from the current frame and return smoothed output.

        Returns:
            (smoothed_label: str, mean_confidence: float)
        """
        # Only valid, above-threshold predictions enter the smoothing window
        if prediction.is_valid:
            self.window.append((prediction.label, prediction.confidence))
        else:
            # If invalid or low confidence, append sentinel
            self.window.append(("...", 0.0))

        # Count frequencies in the window
        valid_labels = [lbl for lbl, _ in self.window if lbl != "..."]

        if not valid_labels:
            return "...", 0.0

        counter = Counter(valid_labels)
        most_common_label, count = counter.most_common(1)[0]
        agreement_ratio = count / len(self.window)

        if agreement_ratio >= self.min_agreement_ratio:
            # Calculate average confidence for the winning label
            confs = [conf for lbl, conf in self.window if lbl == most_common_label]
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            return most_common_label, mean_conf
        else:
            return "...", 0.0

    def reset(self) -> None:
        """Clear the sliding window history."""
        self.window.clear()


class TextAccumulator:
    """
    Sentence Formation Prototype.
    Accumulates confirmed stable signs into a sentence buffer.
    Features:
      - Consecutive frame confirmation threshold
      - Time-based debouncing to prevent accidental repeating
      - Space, Backspace, and Clear operations
    """

    def __init__(
        self,
        confirmation_frames: int = config.CONSECUTIVE_FRAMES_TO_ADD,
        debounce_seconds: float = config.DEBOUNCE_SECONDS,
    ):
        self.confirmation_frames = confirmation_frames
        self.debounce_seconds = debounce_seconds

        self.tokens: List[str] = []
        self._current_candidate: Optional[str] = None
        self._consecutive_count: int = 0
        self._last_added_sign: Optional[str] = None
        self._last_add_time: float = 0.0

    def update(self, stable_sign: str) -> Optional[str]:
        """
        Update the accumulator with the current temporally-smoothed sign.

        Returns:
            The sign token that was newly committed to the sentence, or None.
        """
        if stable_sign == "..." or not stable_sign:
            self._current_candidate = None
            self._consecutive_count = 0
            return None

        current_time = time.time()

        if stable_sign == self._current_candidate:
            self._consecutive_count += 1
        else:
            self._current_candidate = stable_sign
            self._consecutive_count = 1

        # Check if confirmed
        if self._consecutive_count >= self.confirmation_frames:
            # Check debounce condition: don't immediately repeat the exact same sign
            can_add = True
            if (
                stable_sign == self._last_added_sign
                and (current_time - self._last_add_time) < self.debounce_seconds
            ):
                can_add = False

            if can_add:
                self.tokens.append(stable_sign)
                self._last_added_sign = stable_sign
                self._last_add_time = current_time
                self._consecutive_count = 0  # Reset counter after successful commit
                return stable_sign

        return None

    def add_space(self) -> None:
        """Insert a whitespace separation into the sentence."""
        if self.tokens and self.tokens[-1] != " ":
            self.tokens.append(" ")
            self._last_added_sign = None

    def backspace(self) -> None:
        """Remove the last committed sign or space."""
        if self.tokens:
            removed = self.tokens.pop()
            self._last_added_sign = None

    def clear(self) -> None:
        """Wipe the entire sentence buffer."""
        self.tokens.clear()
        self._current_candidate = None
        self._consecutive_count = 0
        self._last_added_sign = None

    def get_text(self) -> str:
        """Return formatted sentence text."""
        # Join tokens cleanly
        text = "".join(
            tok if tok == " " else f" {tok}" for tok in self.tokens
        ).strip()
        return text if text else "..."


if __name__ == "__main__":
    print("=" * 65)
    print("  SignBridge AI - Temporal Smoother & Text Accumulator Test")
    print("=" * 65)

    smoother = TemporalSmoother(window_size=5, min_agreement_ratio=0.6)
    accumulator = TextAccumulator(confirmation_frames=3, debounce_seconds=0.5)

    # Test flicker elimination: ['HELLO', 'HELP', 'HELLO', 'HELLO', 'HELLO']
    test_stream = [
        PredictionResult("HELLO", "HELLO", 0.95, True, {}),
        PredictionResult("HELP", "HELP", 0.88, True, {}),
        PredictionResult("HELLO", "HELLO", 0.94, True, {}),
        PredictionResult("HELLO", "HELLO", 0.96, True, {}),
        PredictionResult("HELLO", "HELLO", 0.93, True, {}),
    ]

    for i, pred in enumerate(test_stream):
        smoothed, conf = smoother.update(pred)
        committed = accumulator.update(smoothed)
        print(f"  Frame {i+1}: Raw='{pred.label}' -> Smoothed='{smoothed}' ({conf:.0%}) | Committed='{committed}'")

    print(f"\nFinal Accumulated Sentence: \"{accumulator.get_text()}\"")
    assert accumulator.get_text() == "HELLO", "Accumulator should have committed 'HELLO'"
    print("[Verified] Temporal smoothing and sentence accumulator functioning as designed.")
