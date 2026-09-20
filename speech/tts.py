"""
SignBridge AI - Offline Text-to-Speech (TTS) Engine (Week 5)
Provides non-blocking speech synthesis for finalized natural language sentences.
Supports English and Hindi (if voice installed on host OS) using pyttsx3.
"""

import sys
import queue
import threading
from pathlib import Path
from typing import Optional, List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.utils.logger import setup_logger

logger = setup_logger("TTS")


class TextToSpeechEngine:
    """
    Thread-safe, non-blocking offline Text-to-Speech engine.
    Ensures speech synthesis does not degrade real-time video FPS.
    """

    def __init__(
        self,
        default_language: str = "en",
        rate: int = 150,
        volume: float = 1.0,
        auto_speak: bool = False,
    ):
        self.default_language = default_language
        self.rate = rate
        self.volume = volume
        self.auto_speak = auto_speak

        self._engine = None
        self._is_available = False
        self._speech_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_speaking = False
        self._voice_map: Dict[str, str] = {}

        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize underlying pyttsx3 synthesis engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            self._discover_voices()
            self._is_available = True
            logger.info("Text-to-Speech engine initialized successfully (offline pyttsx3).")

            # Start background worker thread
            self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self._worker_thread.start()
        except Exception as err:
            logger.warning(f"Could not initialize offline TTS engine: {err}. Speech output disabled.")
            self._engine = None
            self._is_available = False

    def _discover_voices(self) -> None:
        """Enumerate installed system voices and map languages."""
        if self._engine is None:
            return
        try:
            voices = self._engine.getProperty("voices")
            for v in voices:
                name_lower = v.name.lower()
                id_lower = v.id.lower()
                # Check for Hindi voices
                if "hindi" in name_lower or "hi-in" in id_lower or "kalpana" in name_lower or "hemant" in name_lower:
                    self._voice_map["hi"] = v.id
                # Check for English voices
                elif "english" in name_lower or "en-us" in id_lower or "zira" in name_lower or "david" in name_lower:
                    if "en" not in self._voice_map:
                        self._voice_map["en"] = v.id
        except Exception as err:
            logger.warning(f"Voice enumeration warning: {err}")

    def _speech_worker(self) -> None:
        """Background thread consuming speech requests without blocking webcam inference."""
        while not self._stop_event.is_set():
            try:
                item = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            text, lang = item
            self._is_speaking = True
            try:
                if self._engine is not None:
                    # Select voice if available
                    if lang in self._voice_map:
                        self._engine.setProperty("voice", self._voice_map[lang])
                    self._engine.say(text)
                    self._engine.runAndWait()
            except Exception as err:
                logger.error(f"TTS speech execution error: {err}")
            finally:
                self._is_speaking = False
                self._speech_queue.task_done()

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def speak(self, text: str, language: Optional[str] = None) -> bool:
        """
        Enqueues text for asynchronous non-blocking speech playback.

        Args:
            text: Sentence string to pronounce.
            language: 'en' or 'hi' (defaults to default_language).

        Returns:
            True if enqueued, False if TTS unavailable or text empty.
        """
        if not self._is_available:
            return False

        clean_text = text.strip()
        if not clean_text or clean_text == "...":
            return False

        lang = language or self.default_language
        self._speech_queue.put((clean_text, lang))
        logger.info(f"Queued speech: \"{clean_text}\" (lang={lang})")
        return True

    def stop(self) -> None:
        """Interrupt active speech synthesis."""
        try:
            # Drain queue
            while not self._speech_queue.empty():
                try:
                    self._speech_queue.get_nowait()
                    self._speech_queue.task_done()
                except queue.Empty:
                    break
            if self._engine is not None:
                self._engine.stop()
        except Exception as err:
            logger.warning(f"TTS stop warning: {err}")

    def shutdown(self) -> None:
        """Stop worker thread cleanly."""
        self._stop_event.set()
        self._speech_queue.put(None)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
