"""
SignBridge AI - Speech API
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.services.tts_service import tts_service

router = APIRouter(prefix="/speech", tags=["Speech Services"])


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize to speech")
    language: str = Field(default="en", description="Language ('en' or 'hi')")


@router.post("/synthesize")
def synthesize_speech(request: SynthesizeRequest):
    """Triggers server-side TTS speech playback using the offline pyttsx3 engine."""
    success = tts_service.speak(text=request.text, language=request.language)
    return {
        "status": "queued" if success else "unsupported",
        "text": request.text,
        "language": request.language,
        "server_tts_available": tts_service.is_available,
        "note": "Web clients can also leverage native browser window.speechSynthesis directly.",
    }


@router.get("/languages")
def get_supported_languages():
    """Returns languages supported by speech synthesis and translation modules."""
    return {
        "languages": tts_service.get_supported_languages(),
        "default": "en",
    }
