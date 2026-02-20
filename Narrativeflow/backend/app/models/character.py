"""
Character Model - Story characters with detailed profiles
"""
from sqlalchemy import Column, String, Text, Boolean, DateTime, Integer, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class CharacterRole(str, enum.Enum):
    """Character role in story"""
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    DEUTERAGONIST = "deuteragonist"  # Secondary protagonist
    SUPPORTING = "supporting"
    MINOR = "minor"
    MENTOR = "mentor"
    LOVE_INTEREST = "love_interest"
    SIDEKICK = "sidekick"
    FOIL = "foil"
    NARRATOR = "narrator"


class CharacterStatus(str, enum.Enum):
    """Character status in story"""
    ALIVE = "alive"
    DECEASED = "deceased"
    UNKNOWN = "unknown"
    TRANSFORMED = "transformed"


class Character(Base):
    """Character model with comprehensive profile"""
    __tablename__ = "characters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id", ondelete="CASCADE"), nullable=False)
    
    # Basic identity
    name = Column(String(200), nullable=False)
    full_name = Column(String(500), nullable=True)
    aliases = Column(JSONB, default=list)  # Nicknames, titles, etc.
    
    # Role and importance
    role = Column(Enum(CharacterRole), default=CharacterRole.SUPPORTING)
    importance = Column(Integer, default=5)  # 1-10 scale
    status = Column(Enum(CharacterStatus), default=CharacterStatus.ALIVE)
    
    # Demographics
    age = Column(String(50), nullable=True)  # Can be "mid-30s", "immortal", etc.
    gender = Column(String(50), nullable=True)
    species = Column(String(100), default="human")
    occupation = Column(String(200), nullable=True)
    
    # Physical description
    physical_description = Column(Text, nullable=True)
    height = Column(String(50), nullable=True)
    distinguishing_features = Column(JSONB, default=list)
    
    # Personality
    personality_summary = Column(Text, nullable=True)
    personality_traits = Column(JSONB, default=list)
    strengths = Column(JSONB, default=list)
    weaknesses = Column(JSONB, default=list)
    fears = Column(JSONB, default=list)
    desires = Column(JSONB, default=list)
    
    # Background
    backstory = Column(Text, nullable=True)
    motivation = Column(Text, nullable=True)
    internal_conflict = Column(Text, nullable=True)
    external_conflict = Column(Text, nullable=True)
    
    # Voice and speech
    speaking_style = Column(Text, nullable=True)  # How they talk
    catchphrases = Column(JSONB, default=list)
    vocabulary_level = Column(String(50), nullable=True)  # Simple, sophisticated, etc.
    
    # Relationships with other characters
    relationships = Column(JSONB, default=list)  # [{character_id, type, description}]
    
    # Character arc
    arc_description = Column(Text, nullable=True)
    arc_start_state = Column(Text, nullable=True)
    arc_end_state = Column(Text, nullable=True)
    key_moments = Column(JSONB, default=list)
    
    # Current state tracking (updated as story progresses)
    current_emotional_state = Column(String(200), nullable=True)
    current_location = Column(String(200), nullable=True)
    current_goals = Column(JSONB, default=list)
    knowledge = Column(JSONB, default=list)  # What the character knows
    
    # Appearances
    first_appearance_chapter = Column(Integer, nullable=True)
    chapter_appearances = Column(JSONB, default=list)  # List of chapter IDs
    
    # Visual
    portrait_url = Column(Text, nullable=True)
    portrait_prompt = Column(Text, nullable=True)  # For image generation
    image_generation_seed = Column(Integer, nullable=True)  # Seed for consistent image generation
    visual_style = Column(String(100), nullable=True)  # e.g., "anime", "photorealistic", "fantasy art"
    reference_images = Column(JSONB, default=list)  # URLs of reference images for this character
    
    # AI guidance
    ai_writing_notes = Column(Text, nullable=True)  # Notes for AI on how to write this character
    voice_consistency_rules = Column(JSONB, default=list)
    
    # Metadata
    tags = Column(JSONB, default=list)
    custom_fields = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    story = relationship("Story", back_populates="characters")

    def __repr__(self):
        return f"<Character '{self.name}' ({self.role.value})>"
    
    def get_profile_summary(self):
        """Generate a summary for AI context"""
        summary = f"{self.name}"
        if self.role:
            summary += f" ({self.role.value})"
        if self.personality_summary:
            summary += f": {self.personality_summary[:200]}"
        return summary
