"""
SignBridge AI - Health & Readiness API Endpoints
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.session import get_db
from backend.app.services.recognition_service import recognition_service

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/health")
def health_check():
    """Liveness probe: verifies that the service process is up and responding."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
def readiness_check(db: Session = Depends(get_db)):
    """Readiness probe: validates database connectivity and ML model availability."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    static_ready = recognition_service.is_static_available
    dynamic_ready = recognition_service.is_dynamic_available

    all_ready = db_ok and static_ready

    payload = {
        "status": "ready" if all_ready else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_ok,
        "static_model": static_ready,
        "dynamic_model": dynamic_ready,
        "supported_classes_count": len(recognition_service.supported_classes),
    }

    status_code = status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=payload)
