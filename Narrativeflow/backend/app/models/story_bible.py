"""
Story Bible Model - World building and story rules
"""
from sqlalchemy import Column, String, Text, Boolean, DateTime, Integer, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class RuleCategory(str, enum.Enum):
    """Category of world rules"""
    PHYSICS = "physics"  # Laws of physics/magic
    SOCIETY = "society"  # Social rules and norms
    TECHNOLOGY = "technology"
    MAGIC = "magic"
    BIOLOGY = "biology"
    GEOGRAPHY = "geography"
    HISTORY = "history"
    LANGUAGE = "language"
    RELIGION = "religion"
    ECONOMY = "economy"
    POLITICS = "politics"
    CUSTOM = "custom"


class StoryBible(Base):
    """Story Bible - centralized world-building document"""
    __tablename__ = "story_bibles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    # World Overview
    world_name = Column(String(200), nullable=True)
    world_description = Column(Text, nullable=True)
    world_type = Column(String(100), nullable=True)  # e.g., "fantasy medieval", "cyberpunk"
    
    # Setting
    time_period = Column(String(200), nullable=True)
    primary_locations = Column(JSONB, default=list)  # [{name, description, importance}]
    geography = Column(Text, nullable=True)
    climate = Column(Text, nullable=True)
    
    # Society and Culture
    societies = Column(JSONB, default=list)  # [{name, description, customs}]
    social_structure = Column(Text, nullable=True)
    governments = Column(JSONB, default=list)
    religions = Column(JSONB, default=list)
    languages = Column(JSONB, default=list)
    
    # Technology and Magic
    technology_level = Column(String(200), nullable=True)
    technology_description = Column(Text, nullable=True)
    magic_system = Column(Text, nullable=True)
    magic_rules = Column(JSONB, default=list)
    magic_limitations = Column(JSONB, default=list)
    
    # History
    historical_events = Column(JSONB, default=list)  # [{event, date, importance}]
    timeline = Column(JSONB, default=list)  # [{date, event}]
    
    # Factions and Organizations
    factions = Column(JSONB, default=list)  # [{name, description, goals, allies, enemies}]
    
    # Flora and Fauna
    creatures = Column(JSONB, default=list)  # [{name, description, danger_level}]
    plants = Column(JSONB, default=list)
    
    # Items and Artifacts
    important_items = Column(JSONB, default=list)  # [{name, description, powers, location}]
    
    # Themes and Motifs
    central_themes = Column(JSONB, default=list)
    recurring_motifs = Column(JSONB, default=list)
    symbolism = Column(JSONB, default=list)
    
    # Writing Rules
    narrative_rules = Column(JSONB, default=list)  # Rules for narrative consistency
    forbidden_elements = Column(JSONB, default=list)  # Things to avoid
    required_elements = Column(JSONB, default=list)  # Things to include
    
    # Style Guide
    tone_guidelines = Column(Text, nullable=True)
    vocabulary_restrictions = Column(JSONB, default=list)
    style_examples = Column(JSONB, default=list)  # Example passages
    
    # Quick Reference
    quick_facts = Column(JSONB, default=list)  # Fast lookup facts
    glossary = Column(JSONB, default=dict)  # Term definitions
    
    # AI Instructions
    ai_world_context = Column(Text, nullable=True)  # Condensed context for AI
    ai_rules_summary = Column(Text, nullable=True)  # Summary of rules for AI
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    story = relationship("Story", back_populates="story_bible")
    world_rules = relationship("WorldRule", back_populates="story_bible", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<StoryBible for story {self.story_id}>"


class WorldRule(Base):
    """Individual world rule for consistency enforcement"""
    __tablename__ = "world_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_bible_id = Column(UUID(as_uuid=True), ForeignKey("story_bibles.id", ondelete="CASCADE"), nullable=False)
    
    # Rule definition
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(Enum(RuleCategory), default=RuleCategory.CUSTOM)
    
    # Enforcement
    is_strict = Column(Boolean, default=True)  # Must never be violated
    importance = Column(Integer, default=5)  # 1-10
    
    # Examples
    examples = Column(JSONB, default=list)  # [{valid: bool, example: str}]
    exceptions = Column(JSONB, default=list)  # When the rule doesn't apply
    
    # Tracking
    related_chapters = Column(JSONB, default=list)  # Chapters where this rule is relevant
    violations = Column(JSONB, default=list)  # [{chapter_id, description, resolved}]
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    story_bible = relationship("StoryBible", back_populates="world_rules")

    def __repr__(self):
        return f"<WorldRule '{self.title}'>"
