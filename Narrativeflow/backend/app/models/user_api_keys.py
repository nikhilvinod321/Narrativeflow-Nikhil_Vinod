"""
UserApiKeys Model - Per-user external AI provider API keys
Supports: openai, anthropic, gemini
"""
from sqlalchemy import Column, String, Boolean, DateTime, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base


class UserApiKeys(Base):
    """Stores per-user API keys for external AI providers.
    One row per (user, provider). is_active=True marks the currently used provider.
    """
    __tablename__ = "user_api_keys"
    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_user_provider"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    provider = Column(String(32), nullable=False)          # openai | anthropic | gemini
    api_key = Column(Text, nullable=False)                  # Stored as-is (encrypt in production)
    preferred_model = Column(String(128), nullable=True)    # e.g. gpt-4o, claude-3-5-sonnet-latest
    is_active = Column(Boolean, default=False)              # True = this provider is chosen

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<UserApiKeys user={self.user_id} provider={self.provider} active={self.is_active}>"
