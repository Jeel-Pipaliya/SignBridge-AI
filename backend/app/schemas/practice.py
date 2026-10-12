"""
SignBridge AI - Practice Session Schemas
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class PracticeSessionCreate(BaseModel):
    target_sign: str
    predicted_sign: Optional[str] = None
    confidence: Optional[float] = None
    accuracy_score: float = Field(..., ge=0.0, le=1.0)
    is_correct: bool
    notes: Optional[str] = None


class PracticeSessionResponse(BaseModel):
    id: int
    user_id: int
    target_sign: str
    predicted_sign: Optional[str]
    confidence: Optional[float]
    accuracy_score: float
    is_correct: bool
    notes: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class PracticeSummaryResponse(BaseModel):
    total_sessions: int
    correct_count: int
    overall_accuracy: float
    current_streak: int
    most_practiced_signs: List[dict]
