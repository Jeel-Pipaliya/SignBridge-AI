"""
SignBridge AI - Model Metadata & Health Endpoints
"""

from fastapi import APIRouter
from backend.app.services.recognition_service import recognition_service

router = APIRouter(prefix="/model", tags=["AI/ML Model Metadata"])


@router.get("/info")
def get_model_information():
    """Returns comprehensive evaluation and architectural metadata of the real AI models."""
    return recognition_service.get_metadata()


@router.get("/classes")
def get_model_classes():
    """Returns class labels currently supported by the active models."""
    return {
        "static_classes": recognition_service.supported_classes,
        "static_count": len(recognition_service.supported_classes),
        "dynamic_classes": recognition_service.dynamic_classes,
        "dynamic_count": len(recognition_service.dynamic_classes),
    }


@router.get("/status")
def get_model_status():
    """Returns runtime readiness flags of active inference engines."""
    return {
        "static_model_loaded": recognition_service.is_static_available,
        "dynamic_model_loaded": recognition_service.is_dynamic_available,
        "vocabulary_count": len(recognition_service.supported_classes),
    }
