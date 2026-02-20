"""
Plotline Model - Story plotlines and arcs tracking
"""
from sqlalchemy import Column, String, Text, Boolean, DateTime, Integer, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class PlotlineType(str, enum.Enum):
    """Type of plotline"""
    MAIN = "main"
    SUBPLOT = "subplot"
    CHARACTER_ARC = "character_arc"
    ROMANCE = "romance"
    MYSTERY = "mystery"
    CONFLICT = "conflict"
    THEME = "theme"


class PlotlineStatus(str, enum.Enum):
    """Status of plotline"""
    PLANNED = "planned"
    INTRODUCED = "introduced"
    DEVELOPING = "developing"
    CLIMAX = "climax"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class Plotline(Base):
    """Plotline/story arc tracking model"""
    __tablename__ = "plotlines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id", ondelete="CASCADE"), nullable=False)
    
    # Basic info
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    
    # Classification
    type = Column(Enum(PlotlineType), default=PlotlineType.SUBPLOT)
    status = Column(Enum(PlotlineStatus), default=PlotlineStatus.PLANNED)
    importance = Column(Integer, default=5)  # 1-10 scale
    
    # Structure
    setup = Column(Text, nullable=True)  # How it's introduced
    development = Column(Text, nullable=True)  # How it progresses
    climax = Column(Text, nullable=True)  # The peak moment
    resolution = Column(Text, nullable=True)  # How it concludes
    
    # Key events
    key_events = Column(JSONB, default=list)  # [{chapter_id, event_description, order}]
    plot_points = Column(JSONB, default=list)  # Major plot points
    
    # Character involvement
    primary_characters = Column(JSONB, default=list)  # Character IDs
    secondary_characters = Column(JSONB, default=list)
    
    # Chapter tracking
    introduction_chapter = Column(Integer, nullable=True)
    resolution_chapter = Column(Integer, nullable=True)
    chapters_involved = Column(JSONB, default=list)  # Chapter IDs where this plotline is active
    
    # Connections to other plotlines
    connected_plotlines = Column(JSONB, default=list)  # Related plotline IDs
    
    # Foreshadowing and payoffs
    foreshadowing = Column(JSONB, default=list)  # [{chapter, element, description}]
    payoffs = Column(JSONB, default=list)  # [{chapter, element, description}]
    
    # Unresolved threads
    open_questions = Column(JSONB, default=list)  # Things that need resolution
    planted_seeds = Column(JSONB, default=list)  # Setup for future events
    
    # AI notes
    ai_suggestions = Column(JSONB, default=list)  # AI suggestions for this plotline
    consistency_notes = Column(Text, nullable=True)
    
    # Metadata
    color = Column(String(20), nullable=True)  # For UI visualization
    tags = Column(JSONB, default=list)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    story = relationship("Story", back_populates="plotlines")

    def __repr__(self):
        return f"<Plotline '{self.title}' ({self.type.value})>"
    
    @property
    def completion_percentage(self):
        """Estimate completion based on status"""
        status_progress = {
            PlotlineStatus.PLANNED: 0,
            PlotlineStatus.INTRODUCED: 20,
            PlotlineStatus.DEVELOPING: 50,
            PlotlineStatus.CLIMAX: 80,
            PlotlineStatus.RESOLVED: 100,
            PlotlineStatus.ABANDONED: 0
        }
        return status_progress.get(self.status, 0)
