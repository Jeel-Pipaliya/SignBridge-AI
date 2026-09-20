"""
SignBridge AI - Unit Tests for Text-to-Speech Engine (Week 5)
Tests:
  - TextToSpeechEngine non-blocking queuing
  - Voice mapping and language selection
  - Stop and shutdown operations
  - Headless/mock audio verification without hardware dependency
"""

import unittest
from unittest.mock import MagicMock, patch
import time

from speech.tts import TextToSpeechEngine


class TestTextToSpeech(unittest.TestCase):
    """Test suite for TTS engine."""

    @patch("pyttsx3.init")
    def test_tts_initialization_and_queuing(self, mock_pyttsx_init):
        mock_engine = MagicMock()
        mock_pyttsx_init.return_value = mock_engine

        # Mock voice list
        mock_voice_en = MagicMock()
        mock_voice_en.id = "voice_en_david"
        mock_voice_en.name = "Microsoft David Desktop - English (United States)"

        mock_voice_hi = MagicMock()
        mock_voice_hi.id = "voice_hi_kalpana"
        mock_voice_hi.name = "Microsoft Kalpana - Hindi (India)"

        mock_engine.getProperty.return_value = [mock_voice_en, mock_voice_hi]

        tts = TextToSpeechEngine(default_language="en", rate=160, volume=0.9)
        self.assertTrue(tts.is_available)
        self.assertEqual(tts._voice_map.get("en"), "voice_en_david")
        self.assertEqual(tts._voice_map.get("hi"), "voice_hi_kalpana")

        # Test queueing speech
        enqueued = tts.speak("Hello world", language="en")
        self.assertTrue(enqueued)

        # Allow worker thread a brief tick to process
        time.sleep(0.3)
        mock_engine.say.assert_called_with("Hello world")

        # Shutdown cleanly
        tts.shutdown()
        self.assertFalse(tts._worker_thread.is_alive())

    def test_tts_empty_string_rejected(self):
        tts = TextToSpeechEngine()
        self.assertFalse(tts.speak(""))
        self.assertFalse(tts.speak("..."))
        tts.shutdown()


if __name__ == "__main__":
    unittest.main()
