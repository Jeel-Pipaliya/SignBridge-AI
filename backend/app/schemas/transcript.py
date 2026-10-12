"""
SignBridge AI - Transcript Schemas
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class TranscriptCreate(BaseModel):
    title: str = "Live Conversation"
    content: str
    mode: str = "conversation"  # conversation, meeting, emergency
    language: str = "en"


class TranscriptResponse(BaseModel):
    id: int
    user_id: Optional[int]
    title: str
    content: str
    mode: str
    language: str
    created_at: datetime

    class Config:
        from_attributes = True
