"""
SignBridge AI - Text-to-Speech Service
"""

import logging
from typing import Dict, List, Optional
from speech.tts import TextToSpeechEngine

logger = logging.getLogger("TTSService")


class TTSService:
    def __init__(self):
        self._engine: Optional[TextToSpeechEngine] = None
        self._available = False
        try:
            self._engine = TextToSpeechEngine(default_language="en", rate=160, volume=0.95)
            self._available = self._engine.is_available
        except Exception as err:
            logger.warning(f"Could not initialize offline pyttsx3: {err}")

    @property
    def is_available(self) -> bool:
        return self._available

    def speak(self, text: str, language: str = "en") -> bool:
        if self._engine and self._available:
            return self._engine.speak(text, language=language)
        return False

    def get_supported_languages(self) -> List[Dict[str, str]]:
        return [
            {"code": "en", "name": "English (US / India)", "tts_supported": True},
            {"code": "hi", "name": "Hindi (Devanagari / Romanized)", "tts_supported": True},
        ]


tts_service = TTSService()
