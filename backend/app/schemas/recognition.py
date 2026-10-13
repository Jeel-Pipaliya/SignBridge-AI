"""
SignBridge AI - Recognition & Inference Schemas
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0


class LandmarkPredictRequest(BaseModel):
    landmarks: Union[List[LandmarkPoint], List[List[float]], List[float]] = Field(
        ...,
        description="21 3D hand landmarks or flat 63-element coordinate vector"
    )
    handedness: Optional[str] = "Right"
    confidence_threshold: Optional[float] = None
    target_sign: Optional[str] = Field(None, description="Optional target ISL sign for tutor landmark similarity evaluation")


class SequencePredictRequest(BaseModel):
    sequence: List[Union[List[LandmarkPoint], List[List[float]], List[float]]] = Field(
        ...,
        description="Temporal sequence of hand landmark frames (e.g., 30 frames x 21 points / 63 features)"
    )
    confidence_threshold: Optional[float] = None
    target_sign: Optional[str] = None


class CandidatePrediction(BaseModel):
    sign: str
    confidence: float


class FramePredictRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded JPEG or PNG image frame")
    confidence_threshold: Optional[float] = None
    target_sign: Optional[str] = Field(None, description="Optional target ISL sign for tutor landmark similarity evaluation")


class PredictionResponse(BaseModel):
    predicted_sign: str
    confidence: float
    is_recognized: bool
    is_supported_vocabulary: bool
    model_name: str
    model_version: str
    top_candidates: List[CandidatePrediction] = []
    inference_duration_ms: float
    handedness: Optional[str] = None
    hand_detected: bool = True
    landmarks: Optional[List[LandmarkPoint]] = None
    target_sign: Optional[str] = None
    target_similarity: Optional[float] = None
    is_target_match: Optional[bool] = None
    feedback_message: Optional[str] = None


class GrammarCorrectRequest(BaseModel):
    tokens: List[str] = Field(..., description="Ordered list of recognized sign tokens")
    target_language: str = Field(default="en", description="'en' for English, 'hi' for Hindi")


class GrammarCorrectResponse(BaseModel):
    raw_tokens: List[str]
    raw_sentence: str
    corrected_sentence: str
    language: str
    translated_sentence: Optional[str] = None


class CorrectionFeedbackRequest(BaseModel):
    detected_sign: str
    corrected_sign: str
    confidence: Optional[float] = None
    event_id: Optional[int] = None
