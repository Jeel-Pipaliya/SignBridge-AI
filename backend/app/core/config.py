"""
SignBridge AI - Core Configuration Settings
"""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "SignBridge AI"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "signbridge-ai-super-secret-production-key-change-in-prod-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Database
    # Defaults to local SQLite fallback if DATABASE_URL is not set
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./signbridge.db")

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # Model and Asset Paths (relative to repository root)
    STATIC_MODEL_PATH: str = os.getenv(
        "STATIC_MODEL_PATH",
        "ai/models/best/isl_classifier.pkl" if os.path.exists("ai/models/best/isl_classifier.pkl") else "models/best/isl_classifier.pkl",
    )
    STATIC_METADATA_PATH: str = os.getenv(
        "STATIC_METADATA_PATH",
        "ai/models/metadata/model_metadata.json" if os.path.exists("ai/models/metadata/model_metadata.json") else "models/metadata/model_metadata.json",
    )
    DYNAMIC_MODEL_PATH: str = os.getenv(
        "DYNAMIC_MODEL_PATH",
        "ai/models/dynamic/bilstm_dynamic.pt" if os.path.exists("ai/models/dynamic/bilstm_dynamic.pt") else "models/dynamic/bilstm_dynamic.pt",
    )
    DYNAMIC_METADATA_PATH: str = os.getenv(
        "DYNAMIC_METADATA_PATH",
        "ai/models/metadata/dynamic_model_metadata.json" if os.path.exists("ai/models/metadata/dynamic_model_metadata.json") else "models/metadata/dynamic_model_metadata.json",
    )

    # Confidence Threshold Default
    CONFIDENCE_THRESHOLD: float = 0.70

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
