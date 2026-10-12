"""
SignBridge AI - Vocabulary Schemas
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class VocabularyBase(BaseModel):
    sign_name: str
    category: str
    description: Optional[str] = None
    gesture_type: str = "static"
    difficulty: str = "beginner"
    thumbnail_url: Optional[str] = None
    anatomical_tips: Optional[str] = None


class VocabularyResponse(VocabularyBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class VocabularyListResponse(BaseModel):
    total: int
    items: List[VocabularyResponse]
