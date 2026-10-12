"""
SignBridge AI - Analytics & Evaluation Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class VocabularyBreakdownItem(BaseModel):
    sign: str
    practice_count: int
    success_rate: float
    category: str


class ModelEvaluationMetrics(BaseModel):
    model_name: str
    version: str
    architecture: str
    num_classes: int
    train_accuracy: Optional[float]
    val_accuracy: Optional[float]
    test_accuracy: Optional[float]
    test_macro_f1: Optional[float]
    confidence_threshold: float
    classes: List[str]


class AnalyticsOverviewResponse(BaseModel):
    total_recognitions: int
    total_practice_sessions: int
    average_practice_accuracy: float
    most_practiced_signs: List[VocabularyBreakdownItem]
    recent_activity_timeline: List[Dict[str, Any]]
    model_benchmarks: Dict[str, Any]
    offline_vs_live_note: str = (
        "Offline test-set accuracy is measured against held-out ground truth datasets (e.g. 66.07% on 32-class ISL). "
        "Live real-world accuracy represents interactive user sessions and practice ratings."
    )
