"""
SignBridge AI - PracticeSession Model
"""

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.app.db.base import Base


class PracticeSession(Base):
    __tablename__ = "practice_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    target_sign = Column(String(64), index=True, nullable=False)
    predicted_sign = Column(String(64), nullable=True)
    confidence = Column(Float, nullable=True)
    accuracy_score = Column(Float, default=0.0, nullable=False)  # 0.0 to 1.0
    is_correct = Column(Boolean, default=False, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True, nullable=False)

    user = relationship("User", back_populates="practice_sessions")
