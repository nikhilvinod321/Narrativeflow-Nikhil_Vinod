"""
User AI Settings Model - Per-user token limits
"""
from sqlalchemy import Column, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base


class UserAiSettings(Base):
    """Per-user AI token limit overrides"""
    __tablename__ = "user_ai_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)

    # Per-feature token limits (nullable = use defaults)
    max_tokens_story_generation = Column(Integer, nullable=True)
    max_tokens_recap = Column(Integer, nullable=True)
    max_tokens_summary = Column(Integer, nullable=True)
    max_tokens_grammar = Column(Integer, nullable=True)
    max_tokens_branching = Column(Integer, nullable=True)
    max_tokens_story_to_image_prompt = Column(Integer, nullable=True)
    max_tokens_image_to_story = Column(Integer, nullable=True)
    max_tokens_character_extraction = Column(Integer, nullable=True)
    max_tokens_rewrite = Column(Integer, nullable=True)
    max_tokens_dialogue = Column(Integer, nullable=True)
    max_tokens_brainstorm = Column(Integer, nullable=True)
    max_tokens_story_bible = Column(Integer, nullable=True)
    max_tokens_story_bible_update = Column(Integer, nullable=True)
    max_tokens_import_story = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="ai_settings")

    def __repr__(self):
        return f"<UserAiSettings {self.user_id}>"
