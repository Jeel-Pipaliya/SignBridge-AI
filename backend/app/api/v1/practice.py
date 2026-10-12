"""
SignBridge AI - Practice Session Management API
"""

from typing import List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.db.session import get_db
from backend.app.models.practice import PracticeSession
from backend.app.models.user import User
from backend.app.schemas.practice import (
    PracticeSessionCreate,
    PracticeSessionResponse,
    PracticeSummaryResponse,
)

router = APIRouter(prefix="/practice", tags=["Sign Learning Tutor & Practice"])


@router.post("/sessions", response_model=PracticeSessionResponse, status_code=status.HTTP_201_CREATED)
def record_practice_session(
    session_in: PracticeSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Logs a completed interactive sign practice session for the user."""
    session = PracticeSession(
        user_id=current_user.id,
        target_sign=session_in.target_sign.upper(),
        predicted_sign=session_in.predicted_sign.upper() if session_in.predicted_sign else None,
        confidence=session_in.confidence,
        accuracy_score=session_in.accuracy_score,
        is_correct=session_in.is_correct,
        notes=session_in.notes,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return PracticeSessionResponse.model_validate(session)


@router.get("/sessions", response_model=List[PracticeSessionResponse])
def get_practice_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves authenticated user's practice session history."""
    sessions = (
        db.query(PracticeSession)
        .filter(PracticeSession.user_id == current_user.id)
        .order_by(desc(PracticeSession.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [PracticeSessionResponse.model_validate(s) for s in sessions]


@router.get("/summary", response_model=PracticeSummaryResponse)
def get_practice_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Calculates overall practice metrics, streak, and top-practiced signs."""
    total = db.query(PracticeSession).filter(PracticeSession.user_id == current_user.id).count()
    correct = (
        db.query(PracticeSession)
        .filter(PracticeSession.user_id == current_user.id, PracticeSession.is_correct == True)
        .count()
    )

    avg_acc = 0.0
    if total > 0:
        avg_res = (
            db.query(func.avg(PracticeSession.accuracy_score))
            .filter(PracticeSession.user_id == current_user.id)
            .scalar()
        )
        avg_acc = float(avg_res or 0.0)

    # Top practiced signs
    top_signs_query = (
        db.query(PracticeSession.target_sign, func.count(PracticeSession.id).label("cnt"))
        .filter(PracticeSession.user_id == current_user.id)
        .group_by(PracticeSession.target_sign)
        .order_by(desc("cnt"))
        .limit(5)
        .all()
    )
    most_practiced = [{"sign": sign, "count": cnt} for sign, cnt in top_signs_query]

    # Calculate current streak (consecutive correct sessions in recent history)
    recent = (
        db.query(PracticeSession.is_correct)
        .filter(PracticeSession.user_id == current_user.id)
        .order_by(desc(PracticeSession.created_at))
        .limit(20)
        .all()
    )
    streak = 0
    for r in recent:
        if r.is_correct:
            streak += 1
        else:
            break

    return PracticeSummaryResponse(
        total_sessions=total,
        correct_count=correct,
        overall_accuracy=round(avg_acc, 4),
        current_streak=streak,
        most_practiced_signs=most_practiced,
    )
