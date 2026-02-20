"""
Embedding Model - Vector embeddings for semantic memory
"""
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base


class StoryEmbedding(Base):
    """Vector embeddings for story content - enables RAG"""
    __tablename__ = "story_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id", ondelete="CASCADE"), nullable=False)
    chapter_id = Column(UUID(as_uuid=True), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=True)
    
    # Content
    content = Column(Text, nullable=False)  # The text chunk
    content_type = Column(String(50), nullable=False)  # chapter, character, plotline, bible, etc.
    
    # Chunk metadata
    chunk_index = Column(Integer, nullable=False)  # Position in sequence
    chunk_size = Column(Integer, nullable=False)  # Number of characters
    start_position = Column(Integer, nullable=True)  # Start position in original text
    end_position = Column(Integer, nullable=True)  # End position in original text
    
    # The embedding vector - stored as array for pgvector compatibility
    # Using 768 dimensions (common for many embedding models)
    # For pgvector, you'd use: embedding = Column(Vector(768))
    embedding = Column(ARRAY(Float), nullable=True)
    embedding_model = Column(String(100), nullable=True)  # Model used to generate embedding
    
    # Semantic metadata
    summary = Column(Text, nullable=True)  # Brief summary of chunk
    key_entities = Column(JSONB, default=list)  # Characters, locations, etc. mentioned
    key_events = Column(JSONB, default=list)  # Events in this chunk
    emotional_tone = Column(String(100), nullable=True)
    
    # Relevance tracking
    retrieval_count = Column(Integer, default=0)  # How often retrieved
    last_retrieved_at = Column(DateTime, nullable=True)
    relevance_score = Column(Float, nullable=True)  # Quality score
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    story = relationship("Story", back_populates="embeddings")
    chapter = relationship("Chapter", back_populates="embeddings")

    def __repr__(self):
        return f"<StoryEmbedding {self.content_type}:{self.chunk_index}>"


class CharacterEmbedding(Base):
    """Embeddings specifically for character information"""
    __tablename__ = "character_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    character_id = Column(UUID(as_uuid=True), ForeignKey("characters.id", ondelete="CASCADE"), nullable=False)
    
    # Content type
    content_type = Column(String(50), nullable=False)  # profile, dialogue, action, relationship
    content = Column(Text, nullable=False)
    
    # Embedding
    embedding = Column(ARRAY(Float), nullable=True)
    
    # Source tracking
    source_chapter_id = Column(UUID(as_uuid=True), nullable=True)
    source_context = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<CharacterEmbedding {self.character_id}:{self.content_type}>"
