"""
SignBridge AI - User & Auth Pydantic Schemas
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="User password, minimum 6 characters")


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserPreferenceBase(BaseModel):
    theme: str = "dark"
    font_size: str = "normal"
    high_contrast: bool = False
    reduced_motion: bool = False
    tts_speed: float = 1.0
    tts_volume: float = 1.0
    preferred_language: str = "en"
    auto_speak: bool = True
    store_history: bool = True


class UserPreferenceUpdate(BaseModel):
    theme: Optional[str] = None
    font_size: Optional[str] = None
    high_contrast: Optional[bool] = None
    reduced_motion: Optional[bool] = None
    tts_speed: Optional[float] = None
    tts_volume: Optional[float] = None
    preferred_language: Optional[str] = None
    auto_speak: Optional[bool] = None
    store_history: Optional[bool] = None


class UserPreferenceResponse(UserPreferenceBase):
    id: int
    user_id: int
    updated_at: datetime

    class Config:
        from_attributes = True


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    preferences: Optional[UserPreferenceResponse] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[int] = None
