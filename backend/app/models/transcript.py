"""
SignBridge AI - TranscriptRecord Model
"""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.app.db.base import Base


class TranscriptRecord(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True)
    title = Column(String(255), default="Untitled Conversation", nullable=False)
    content = Column(Text, nullable=False)
    mode = Column(String(32), default="conversation", nullable=False)  # conversation, meeting, emergency
    language = Column(String(16), default="en", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True, nullable=False)

    user = relationship("User", back_populates="transcripts")
