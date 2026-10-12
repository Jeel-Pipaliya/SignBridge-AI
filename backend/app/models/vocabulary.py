"""
SignBridge AI - VocabularySign Model
"""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String, Text

from backend.app.db.base import Base


class VocabularySign(Base):
    __tablename__ = "vocabulary_signs"

    id = Column(Integer, primary_key=True, index=True)
    sign_name = Column(String(64), unique=True, index=True, nullable=False)
    category = Column(String(64), index=True, nullable=False)  # alphabet, number, greeting, phrase
    description = Column(Text, nullable=True)
    gesture_type = Column(String(32), default="static", nullable=False)  # static, dynamic
    difficulty = Column(String(32), default="beginner", nullable=False)  # beginner, intermediate, advanced
    thumbnail_url = Column(String(512), nullable=True)
    anatomical_tips = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
