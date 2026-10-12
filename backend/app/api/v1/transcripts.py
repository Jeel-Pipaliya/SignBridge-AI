"""
SignBridge AI - Transcripts API
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.api.deps import get_optional_current_user
from backend.app.db.session import get_db
from backend.app.models.transcript import TranscriptRecord
from backend.app.models.user import User
from backend.app.schemas.transcript import TranscriptCreate, TranscriptResponse

router = APIRouter(prefix="/transcripts", tags=["Transcripts & Captions"])


@router.post("", response_model=TranscriptResponse, status_code=status.HTTP_201_CREATED)
def save_transcript(
    transcript_in: TranscriptCreate,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Saves a conversation, meeting, or caption transcript with explicit user consent."""
    user_id = current_user.id if current_user else None
    transcript = TranscriptRecord(
        user_id=user_id,
        title=transcript_in.title,
        content=transcript_in.content,
        mode=transcript_in.mode,
        language=transcript_in.language,
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)
    return TranscriptResponse.model_validate(transcript)


@router.get("", response_model=List[TranscriptResponse])
def list_transcripts(
    mode: Optional[str] = Query(None, description="Filter by mode (conversation, meeting, emergency)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=50),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Lists saved transcripts."""
    query = db.query(TranscriptRecord)
    if current_user:
        query = query.filter(TranscriptRecord.user_id == current_user.id)
    if mode:
        query = query.filter(TranscriptRecord.mode == mode)

    records = query.order_by(desc(TranscriptRecord.created_at)).offset(skip).limit(limit).all()
    return [TranscriptResponse.model_validate(r) for r in records]


@router.get("/{transcript_id}", response_model=TranscriptResponse)
def get_transcript(transcript_id: int, db: Session = Depends(get_db)):
    """Retrieves a single transcript by ID."""
    t = db.query(TranscriptRecord).filter(TranscriptRecord.id == transcript_id).first()
    if not t:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found",
        )
    return TranscriptResponse.model_validate(t)


@router.delete("/{transcript_id}", status_code=status.HTTP_200_OK)
def delete_transcript(
    transcript_id: int,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Deletes a transcript."""
    query = db.query(TranscriptRecord).filter(TranscriptRecord.id == transcript_id)
    if current_user:
        query = query.filter(TranscriptRecord.user_id == current_user.id)
    t = query.first()
    if not t:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found or access denied",
        )
    db.delete(t)
    db.commit()
    return {"status": "success", "deleted_id": transcript_id}
