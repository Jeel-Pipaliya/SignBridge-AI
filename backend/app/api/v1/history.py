"""
SignBridge AI - History & Data Privacy Management API
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.db.session import get_db
from backend.app.models.practice import PracticeSession
from backend.app.models.recognition import RecognitionEvent
from backend.app.models.user import User

router = APIRouter(prefix="/history", tags=["History & Privacy Controls"])


@router.get("/recognition")
def get_recognition_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves user's recorded recognition events."""
    events = (
        db.query(RecognitionEvent)
        .filter(RecognitionEvent.user_id == current_user.id)
        .order_by(desc(RecognitionEvent.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": e.id,
            "sign_detected": e.sign_detected,
            "confidence": e.confidence,
            "model_name": e.model_name,
            "model_version": e.model_version,
            "duration_ms": e.inference_duration_ms,
            "was_corrected": e.was_corrected,
            "corrected_sign": e.corrected_sign,
            "created_at": e.created_at.isoformat(),
        }
        for e in events
    ]


@router.delete("/clear", status_code=status.HTTP_200_OK)
def clear_user_history(
    clear_practice: bool = Query(False, description="Also clear practice history if true"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Privacy-first data erasure: deletes all stored recognition events
    and optionally practice sessions for the authenticated user.
    """
    deleted_recs = (
        db.query(RecognitionEvent)
        .filter(RecognitionEvent.user_id == current_user.id)
        .delete(synchronize_session=False)
    )

    deleted_prac = 0
    if clear_practice:
        deleted_prac = (
            db.query(PracticeSession)
            .filter(PracticeSession.user_id == current_user.id)
            .delete(synchronize_session=False)
        )

    db.commit()

    return {
        "status": "success",
        "message": "User history successfully wiped in compliance with privacy retention settings.",
        "recognition_events_deleted": deleted_recs,
        "practice_sessions_deleted": deleted_prac,
    }
