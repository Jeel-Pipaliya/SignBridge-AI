"""
SignBridge AI - ModelVersion Model
"""

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from backend.app.db.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(128), index=True, nullable=False)
    version = Column(String(32), index=True, nullable=False)
    architecture = Column(String(64), nullable=False)  # MLP, BiLSTM, SVM, etc.
    num_classes = Column(Integer, nullable=False)
    confidence_threshold = Column(Float, default=0.70, nullable=False)
    train_accuracy = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    test_macro_f1 = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
