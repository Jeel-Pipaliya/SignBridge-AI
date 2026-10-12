"""
SignBridge AI - Real-Time Recognition & Inference API
"""

from typing import Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.api.deps import get_optional_current_user
from backend.app.db.session import get_db
from backend.app.models.recognition import RecognitionEvent
from backend.app.models.user import User
from backend.app.schemas.recognition import (
    CorrectionFeedbackRequest,
    GrammarCorrectRequest,
    GrammarCorrectResponse,
    LandmarkPredictRequest,
    PredictionResponse,
    SequencePredictRequest,
)
from backend.app.services.grammar_service import grammar_service
from backend.app.services.recognition_service import recognition_service

router = APIRouter(prefix="/recognition", tags=["Real-Time Recognition"])


@router.post("/predict-landmarks", response_model=PredictionResponse)
def predict_landmarks(
    request: LandmarkPredictRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Sub-millisecond inference for 21 3D hand landmarks or 63-element feature vector.
    Applies invariant wrist translation & scale normalization before classification.
    """
    response = recognition_service.predict_landmarks(
        landmarks=request.landmarks,
        handedness=request.handedness,
        confidence_threshold=request.confidence_threshold,
    )

    # If authenticated user has history persistence enabled, record event
    if current_user and response.is_recognized:
        should_store = True
        if current_user.preferences and not current_user.preferences.store_history:
            should_store = False

        if should_store:
            try:
                event = RecognitionEvent(
                    user_id=current_user.id,
                    sign_detected=response.predicted_sign,
                    confidence=response.confidence,
                    model_name=response.model_name,
                    model_version=response.model_version,
                    inference_duration_ms=response.inference_duration_ms,
                    was_corrected=False,
                )
                db.add(event)
                db.commit()
            except Exception:
                db.rollback()

    return response


@router.post("/predict-sequence", response_model=PredictionResponse)
def predict_sequence(
    request: SequencePredictRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Inference on a 30-frame temporal landmark sequence using the trained Bi-LSTM model.
    Detects dynamic ISL gestures (e.g. HELLO, J, Z).
    """
    response = recognition_service.predict_sequence(
        sequence=request.sequence,
        confidence_threshold=request.confidence_threshold,
    )

    if current_user and response.is_recognized:
        should_store = True
        if current_user.preferences and not current_user.preferences.store_history:
            should_store = False

        if should_store:
            try:
                event = RecognitionEvent(
                    user_id=current_user.id,
                    sign_detected=response.predicted_sign,
                    confidence=response.confidence,
                    model_name=response.model_name,
                    model_version=response.model_version,
                    inference_duration_ms=response.inference_duration_ms,
                    was_corrected=False,
                )
                db.add(event)
                db.commit()
            except Exception:
                db.rollback()

    return response


@router.post("/grammar-correct", response_model=GrammarCorrectResponse)
def grammar_correct(request: GrammarCorrectRequest):
    """
    Reconstructs recognized sign tokens into a grammatical sentence
    and optionally provides Hindi translation.
    """
    return grammar_service.correct_and_translate(
        tokens=request.tokens,
        target_language=request.target_language,
    )


@router.post("/feedback", status_code=status.HTTP_200_OK)
def log_correction_feedback(
    feedback: CorrectionFeedbackRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Records manual user correction when a sign is misrecognized
    to support active learning and evaluation telemetry.
    """
    user_id = current_user.id if current_user else None
    event = RecognitionEvent(
        user_id=user_id,
        sign_detected=feedback.detected_sign,
        confidence=feedback.confidence or 0.0,
        corrected_sign=feedback.corrected_sign,
        was_corrected=True,
    )
    db.add(event)
    db.commit()
    return {
        "status": "feedback_recorded",
        "detected": feedback.detected_sign,
        "corrected_to": feedback.corrected_sign,
    }
