"""
Generated Image Model - Store metadata for AI-generated images
"""
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class ImageType(str, enum.Enum):
    """Type of generated image"""
    CHARACTER = "character"
    SCENE = "scene"
    COVER = "cover"
    ENVIRONMENT = "environment"
    OTHER = "other"


class GeneratedImage(Base):
    """Model for storing generated image metadata"""
    __tablename__ = "generated_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id", ondelete="CASCADE"), nullable=False)
    character_id = Column(UUID(as_uuid=True), ForeignKey("characters.id", ondelete="SET NULL"), nullable=True)
    
    # Image details
    image_type = Column(Enum(ImageType), default=ImageType.SCENE)
    title = Column(String(255), nullable=True)  # User-given title or auto-generated
    description = Column(Text, nullable=True)  # What the image depicts
    
    # File information
    file_path = Column(Text, nullable=False)  # Relative path like /static/generated_images/xxx.png
    file_name = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=True)  # File size in bytes
    
    # Generation details
    prompt = Column(Text, nullable=True)  # The prompt used to generate
    style_id = Column(String(50), nullable=True)  # Art style used (ghibli, anime, etc.)
    seed = Column(Integer, nullable=True)  # Random seed for reproducibility
    
    # Source content
    source_text = Column(Text, nullable=True)  # Original story text used
    
    # Metadata
    tags = Column(JSONB, default=list)  # User tags for organization
    is_favorite = Column(Integer, default=0)  # 0 = not favorite, 1 = favorite
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    story = relationship("Story", back_populates="generated_images")
    character = relationship("Character", backref="generated_images")

    def __repr__(self):
        return f"<GeneratedImage {self.id} ({self.image_type.value})>"
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": str(self.id),
            "story_id": str(self.story_id),
            "character_id": str(self.character_id) if self.character_id else None,
            "image_type": self.image_type.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "prompt": self.prompt,
            "style_id": self.style_id,
            "seed": self.seed,
            "tags": self.tags or [],
            "is_favorite": bool(self.is_favorite),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
