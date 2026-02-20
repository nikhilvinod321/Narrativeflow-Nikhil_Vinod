"""
Generation History Model - Track AI generation history and writing modes
"""
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Float, Enum, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class WritingMode(str, enum.Enum):
    """Writing mode for AI collaboration"""
    AI_LEAD = "ai_lead"  # AI writes autonomously
    USER_LEAD = "user_lead"  # User writes, AI assists
    CO_AUTHOR = "co_author"  # Turn-based collaboration


class GenerationType(str, enum.Enum):
    """Type of AI generation"""
    CONTINUATION = "continuation"
    REWRITE = "rewrite"
    SUMMARY = "summary"
    DIALOGUE = "dialogue"
    DESCRIPTION = "description"
    BRAINSTORM = "brainstorm"
    CONSISTENCY_CHECK = "consistency_check"
    CHARACTER_VOICE = "character_voice"
    PLOT_SUGGESTION = "plot_suggestion"
    IMAGE_PROMPT = "image_prompt"


class GenerationHistory(Base):
    """Track all AI generations for history and learning"""
    __tablename__ = "generation_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id", ondelete="CASCADE"), nullable=False)
    chapter_id = Column(UUID(as_uuid=True), ForeignKey("chapters.id", ondelete="SET NULL"), nullable=True)
    
    # Generation type
    generation_type = Column(Enum(GenerationType), nullable=False)
    writing_mode = Column(Enum(WritingMode), nullable=False)
    
    # Input/Output
    prompt = Column(Text, nullable=False)  # The prompt sent to AI
    system_prompt = Column(Text, nullable=True)  # System instructions
    context_provided = Column(Text, nullable=True)  # Story context given
    
    output = Column(Text, nullable=False)  # AI response
    output_word_count = Column(Integer, nullable=True)
    
    # Quality and feedback
    user_rating = Column(Integer, nullable=True)  # 1-5 stars
    was_accepted = Column(Boolean, nullable=True)  # Did user keep it?
    was_edited = Column(Boolean, nullable=True)  # Did user modify it?
    edited_version = Column(Text, nullable=True)  # What user changed it to
    feedback = Column(Text, nullable=True)  # User feedback
    
    # AI model info
    model_used = Column(String(100), nullable=False)
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    
    # Performance metrics
    generation_time_ms = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    
    # Context retrieval (RAG)
    retrieved_chunks = Column(JSONB, default=list)  # IDs of retrieved embeddings
    retrieval_scores = Column(JSONB, default=list)  # Relevance scores
    
    # Error tracking
    had_error = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    story = relationship("Story", back_populates="generations")

    def __repr__(self):
        return f"<Generation {self.generation_type.value} at {self.created_at}>"


class AISession(Base):
    """Track AI writing sessions"""
    __tablename__ = "ai_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Session info
    writing_mode = Column(Enum(WritingMode), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    
    # Progress
    words_written_by_user = Column(Integer, default=0)
    words_written_by_ai = Column(Integer, default=0)
    generations_count = Column(Integer, default=0)
    
    # Chapters worked on
    chapters_edited = Column(JSONB, default=list)
    
    # Session summary
    session_summary = Column(Text, nullable=True)  # AI summary of what was accomplished

    def __repr__(self):
        return f"<AISession {self.writing_mode.value} started {self.started_at}>"
