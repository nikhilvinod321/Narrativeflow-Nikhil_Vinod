"""
Story Model - Main story/project container
"""
from sqlalchemy import Column, String, Text, Boolean, DateTime, Integer, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class StoryGenre(str, enum.Enum):
    """Story genre types"""
    FANTASY = "fantasy"
    SCIENCE_FICTION = "science_fiction"
    ROMANCE = "romance"
    THRILLER = "thriller"
    MYSTERY = "mystery"
    HORROR = "horror"
    LITERARY = "literary"
    HISTORICAL = "historical"
    ADVENTURE = "adventure"
    DRAMA = "drama"
    COMEDY = "comedy"
    DYSTOPIAN = "dystopian"
    YOUNG_ADULT = "young_adult"
    CRIME = "crime"
    WESTERN = "western"
    OTHER = "other"


class StoryTone(str, enum.Enum):
    """Story tone/mood"""
    DARK = "dark"
    LIGHT = "light"
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    WHIMSICAL = "whimsical"
    GRITTY = "gritty"
    ROMANTIC = "romantic"
    SUSPENSEFUL = "suspenseful"
    MELANCHOLIC = "melancholic"
    HOPEFUL = "hopeful"
    EPIC = "epic"
    INTIMATE = "intimate"


class StoryStatus(str, enum.Enum):
    """Story status"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Story(Base):
    """Main story container model"""
    __tablename__ = "stories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    author_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Basic info
    title = Column(String(500), nullable=False)
    subtitle = Column(String(500), nullable=True)
    logline = Column(Text, nullable=True)  # One-line summary
    synopsis = Column(Text, nullable=True)  # Full synopsis
    
    # Classification
    genre = Column(Enum(StoryGenre), default=StoryGenre.OTHER)
    secondary_genre = Column(Enum(StoryGenre), nullable=True)
    tone = Column(Enum(StoryTone), default=StoryTone.SERIOUS)
    status = Column(Enum(StoryStatus), default=StoryStatus.DRAFT)
    
    # Story settings
    setting_time = Column(Text, nullable=True)  # e.g., "Victorian Era", "2150 AD"
    setting_place = Column(Text, nullable=True)  # e.g., "London", "Mars Colony", or longer descriptions
    pov_style = Column(String(50), default="third_person_limited")  # first_person, third_person_limited, third_person_omniscient
    tense = Column(String(20), default="past")  # past, present
    language = Column(String(50), default="English")  # Story language: English, Japanese, Chinese, Korean, etc.
    
    # AI settings
    writing_style = Column(Text, nullable=True)  # Custom style instructions
    ai_personality = Column(Text, nullable=True)  # AI co-author personality
    creativity_level = Column(Integer, default=7)  # 1-10 scale
    
    # Cover and visuals
    cover_image_url = Column(Text, nullable=True)
    cover_prompt = Column(Text, nullable=True)
    color_theme = Column(String(50), nullable=True)
    
    # Statistics
    word_count = Column(Integer, default=0)
    chapter_count = Column(Integer, default=0)
    
    # Metadata
    tags = Column(JSONB, default=list)
    custom_metadata = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_written_at = Column(DateTime, nullable=True)
    
    # Relationships
    author = relationship("User", back_populates="stories")
    chapters = relationship("Chapter", back_populates="story", cascade="all, delete-orphan", order_by="Chapter.order")
    characters = relationship("Character", back_populates="story", cascade="all, delete-orphan")
    plotlines = relationship("Plotline", back_populates="story", cascade="all, delete-orphan")
    story_bible = relationship("StoryBible", back_populates="story", uselist=False, cascade="all, delete-orphan")
    embeddings = relationship("StoryEmbedding", back_populates="story", cascade="all, delete-orphan")
    generations = relationship("GenerationHistory", back_populates="story", cascade="all, delete-orphan")
    generated_images = relationship("GeneratedImage", back_populates="story", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Story '{self.title}'>"
    
    @property
    def is_complete(self):
        return self.status == StoryStatus.COMPLETED
