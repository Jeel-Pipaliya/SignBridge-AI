"""
SignBridge AI - RecognitionEvent Model
"""

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from backend.app.db.base import Base


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    sign_detected = Column(String(64), index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    model_name = Column(String(64), default="SignBridge_ISL_MLP_NeuralNet", nullable=False)
    model_version = Column(String(32), default="1.0", nullable=False)
    inference_duration_ms = Column(Float, nullable=True)
    corrected_sign = Column(String(64), nullable=True)
    was_corrected = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True, nullable=False)

    user = relationship("User", back_populates="recognition_events")
