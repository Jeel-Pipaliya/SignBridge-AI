"""
SignBridge AI - Vocabulary Management API
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.vocabulary import VocabularySign
from backend.app.schemas.vocabulary import VocabularyListResponse, VocabularyResponse

router = APIRouter(prefix="/vocabulary", tags=["ISL Vocabulary"])


@router.get("", response_model=VocabularyListResponse)
def list_vocabulary(
    category: Optional[str] = Query(None, description="Filter by category (alphabet, number, greeting, phrase)"),
    gesture_type: Optional[str] = Query(None, description="Filter by static or dynamic"),
    search: Optional[str] = Query(None, description="Search by sign name or description"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Retrieves standard Indian Sign Language vocabulary catalog."""
    query = db.query(VocabularySign)
    if category:
        query = query.filter(VocabularySign.category == category.lower())
    if gesture_type:
        query = query.filter(VocabularySign.gesture_type == gesture_type.lower())
    if search:
        query = query.filter(
            VocabularySign.sign_name.ilike(f"%{search}%") | VocabularySign.description.ilike(f"%{search}%")
        )

    total = query.count()
    items = query.order_by(VocabularySign.sign_name.asc()).offset(skip).limit(limit).all()

    return VocabularyListResponse(
        total=total,
        items=[VocabularyResponse.model_validate(item) for item in items],
    )


@router.get("/{sign_id}", response_model=VocabularyResponse)
def get_vocabulary_sign(sign_id: int, db: Session = Depends(get_db)):
    """Retrieves detailed instruction, description, and anatomical tips for a single sign."""
    sign = db.query(VocabularySign).filter(VocabularySign.id == sign_id).first()
    if not sign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vocabulary sign ID {sign_id} not found",
        )
    return VocabularyResponse.model_validate(sign)
