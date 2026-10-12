"""
SignBridge AI - Analytics & Evaluation Telemetry API
"""

from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.db.session import get_db
from backend.app.models.practice import PracticeSession
from backend.app.models.recognition import RecognitionEvent
from backend.app.models.user import User
from backend.app.models.vocabulary import VocabularySign
from backend.app.schemas.analytics import (
    AnalyticsOverviewResponse,
    VocabularyBreakdownItem,
)
from backend.app.services.recognition_service import recognition_service

router = APIRouter(prefix="/analytics", tags=["Analytics & Evaluation"])


@router.get("/overview", response_model=AnalyticsOverviewResponse)
def get_analytics_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns aggregated metrics combining user's live interaction history
    with held-out offline ML model evaluation benchmarks.
    """
    total_recs = (
        db.query(RecognitionEvent)
        .filter(RecognitionEvent.user_id == current_user.id)
        .count()
    )

    total_prac = (
        db.query(PracticeSession)
        .filter(PracticeSession.user_id == current_user.id)
        .count()
    )

    avg_prac_acc = 0.0
    if total_prac > 0:
        avg_res = (
            db.query(func.avg(PracticeSession.accuracy_score))
            .filter(PracticeSession.user_id == current_user.id)
            .scalar()
        )
        avg_prac_acc = round(float(avg_res or 0.0), 4)

    # Vocabulary practice breakdown
    vocab_breakdown: List[VocabularyBreakdownItem] = []
    sign_stats = (
        db.query(
            PracticeSession.target_sign,
            func.count(PracticeSession.id).label("total_cnt"),
            func.sum(case((PracticeSession.is_correct == True, 1), else_=0)).label("correct_cnt"),
        )
        .filter(PracticeSession.user_id == current_user.id)
        .group_by(PracticeSession.target_sign)
        .order_by(desc("total_cnt"))
        .limit(10)
        .all()
    )

    # Category map for signs
    category_map = {
        s.sign_name: s.category
        for s in db.query(VocabularySign).all()
    }

    for row in sign_stats:
        sign_name = row.target_sign
        t_cnt = row.total_cnt or 0
        c_cnt = row.correct_cnt or 0
        rate = round((c_cnt / t_cnt), 4) if t_cnt > 0 else 0.0
        cat = category_map.get(sign_name, "general")
        vocab_breakdown.append(
            VocabularyBreakdownItem(
                sign=sign_name,
                practice_count=t_cnt,
                success_rate=rate,
                category=cat,
            )
        )

    # Recent activity timeline
    recent_sessions = (
        db.query(PracticeSession)
        .filter(PracticeSession.user_id == current_user.id)
        .order_by(desc(PracticeSession.created_at))
        .limit(5)
        .all()
    )
    timeline = [
        {
            "id": s.id,
            "type": "practice",
            "target": s.target_sign,
            "accuracy": s.accuracy_score,
            "is_correct": s.is_correct,
            "timestamp": s.created_at.isoformat(),
        }
        for s in recent_sessions
    ]

    # Offline benchmark statistics from real trained model
    benchmarks = recognition_service.get_metadata()

    return AnalyticsOverviewResponse(
        total_recognitions=total_recs,
        total_practice_sessions=total_prac,
        average_practice_accuracy=avg_prac_acc,
        most_practiced_signs=vocab_breakdown,
        recent_activity_timeline=timeline,
        model_benchmarks=benchmarks,
    )
