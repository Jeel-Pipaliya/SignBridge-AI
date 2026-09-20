"""
SignBridge AI - Unit Tests for Sequence and Token Buffers (Week 5)
Tests:
  - SequenceBuilder accumulation, fixed length, sliding window
  - TokenBuffer static consecutive frame confirmation
  - TokenBuffer dynamic gesture instant confirmation
  - Time-based debounce suppression
  - Backspace, Clear, Space, and Sentence finalization
"""

import unittest
import time
import numpy as np

from ai.preprocessing.sequence_builder import SequenceBuilder
from ai.fusion.token_buffer import TokenBuffer


class TestSequenceAndTokenBuffers(unittest.TestCase):
    """Test suite for sequence buffering and token aggregation."""

    def test_sequence_builder_accumulation(self):
        builder = SequenceBuilder(sequence_length=30)
        self.assertFalse(builder.is_ready())
        self.assertIsNone(builder.get_sequence())

        for i in range(29):
            builder.add_frame(np.zeros(63))
        self.assertFalse(builder.is_ready())

        builder.add_frame(np.zeros(63))
        self.assertTrue(builder.is_ready())
        seq = builder.get_sequence()
        self.assertIsNotNone(seq)
        self.assertEqual(seq.shape, (30, 63))

        # Reset
        builder.reset()
        self.assertFalse(builder.is_ready())

    def test_token_buffer_static_confirmation(self):
        buffer = TokenBuffer(confirmation_frames=3, debounce_seconds=1.0)
        # Frame 1: not confirmed
        self.assertIsNone(buffer.update("A", confidence=0.8, source="STATIC"))
        # Frame 2: not confirmed
        self.assertIsNone(buffer.update("A", confidence=0.8, source="STATIC"))
        # Frame 3: confirmed!
        committed = buffer.update("A", confidence=0.85, source="STATIC")
        self.assertEqual(committed, "A")
        self.assertEqual(buffer.get_tokens(), ["A"])
        self.assertEqual(buffer.get_text(), "A")

    def test_token_buffer_dynamic_confirmation(self):
        buffer = TokenBuffer(confirmation_frames=5, debounce_seconds=1.0)
        # Dynamic gesture commits immediately if valid
        committed = buffer.update("HELLO", confidence=0.92, source="DYNAMIC")
        self.assertEqual(committed, "HELLO")
        self.assertEqual(buffer.get_tokens(), ["HELLO"])

    def test_token_buffer_debounce(self):
        buffer = TokenBuffer(confirmation_frames=1, debounce_seconds=1.0)
        # Commit 1
        committed1 = buffer.update("B", confidence=0.9, source="STATIC")
        self.assertEqual(committed1, "B")

        # Immediate repeat should be blocked by debounce
        committed2 = buffer.update("B", confidence=0.9, source="STATIC")
        self.assertIsNone(committed2)
        self.assertEqual(buffer.get_tokens(), ["B"])

    def test_token_buffer_operations(self):
        buffer = TokenBuffer(confirmation_frames=1, debounce_seconds=0.1)
        buffer.update("I", confidence=0.9, source="STATIC")
        time.sleep(0.15)
        buffer.update("GO", confidence=0.9, source="STATIC")
        time.sleep(0.15)
        buffer.update("HOME", confidence=0.9, source="STATIC")
        self.assertEqual(buffer.get_tokens(), ["I", "GO", "HOME"])
        self.assertEqual(buffer.get_text(), "I GO HOME")

        # Backspace
        removed = buffer.backspace()
        self.assertEqual(removed.token, "HOME")
        self.assertEqual(buffer.get_tokens(), ["I", "GO"])

        # Finalize
        final_text = buffer.finalize_sentence()
        self.assertEqual(final_text, "I GO")
        self.assertTrue(buffer.is_finalized)

        # Clear
        buffer.clear()
        self.assertEqual(buffer.get_tokens(), [])
        self.assertEqual(buffer.get_text(), "")


if __name__ == "__main__":
    unittest.main()
