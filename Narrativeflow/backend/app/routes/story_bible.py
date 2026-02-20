"""
Story Bible Routes - World-building and story rules management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID

from app.database import get_db
from app.models.user import User
from app.models.story_bible import StoryBible, WorldRule, RuleCategory
from app.services.story_service import StoryService
from app.services.memory_service import MemoryService
from app.services.gemini_service import GeminiService
from app.services.chapter_service import ChapterService
from app.services.character_service import CharacterService
from app.services.token_settings import get_user_token_limits
from app.routes.auth import get_current_user
import logging
import traceback

logger = logging.getLogger(__name__)

router = APIRouter()
story_service = StoryService()
memory_service = MemoryService()
gemini_service = GeminiService()
chapter_service = ChapterService()
character_service = CharacterService()


# Pydantic models
class StoryBibleUpdate(BaseModel):
    world_name: Optional[str] = None
    world_description: Optional[str] = None
    world_type: Optional[str] = None
    time_period: Optional[str] = None
    primary_locations: Optional[List[dict]] = None
    geography: Optional[str] = None
    climate: Optional[str] = None
    societies: Optional[List[dict]] = None
    social_structure: Optional[str] = None
    governments: Optional[List[dict]] = None
    religions: Optional[List[dict]] = None
    technology_level: Optional[str] = None
    technology_description: Optional[str] = None
    magic_system: Optional[str] = None
    magic_rules: Optional[List[str]] = None
    magic_limitations: Optional[List[str]] = None
    historical_events: Optional[List[dict]] = None
    factions: Optional[List[dict]] = None
    creatures: Optional[List[dict]] = None
    important_items: Optional[List[dict]] = None
    central_themes: Optional[List[str]] = None
    recurring_motifs: Optional[List[str]] = None
    tone_guidelines: Optional[str] = None
    quick_facts: Optional[List[str]] = None
    glossary: Optional[Any] = None  # Can be dict or list
    # Additional fields from frontend
    world_rules: Optional[List[dict]] = None  # List of {category, rule_text}
    key_locations: Optional[List[dict]] = None  # List of {name, description}
    themes: Optional[List[str]] = None


class WorldRuleCreate(BaseModel):
    title: str
    description: str
    category: RuleCategory = RuleCategory.CUSTOM
    is_strict: bool = True
    importance: int = 5
    examples: List[dict] = []
    exceptions: List[str] = []


class WorldRuleUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[RuleCategory] = None
    is_strict: Optional[bool] = None
    importance: Optional[int] = None
    examples: Optional[List[dict]] = None
    exceptions: Optional[List[str]] = None


class WorldRuleResponse(BaseModel):
    id: UUID
    title: str
    description: str
    category: RuleCategory
    is_strict: bool
    importance: int
    examples: List[dict]
    exceptions: List[str]
    created_at: datetime

    class Config:
        from_attributes = True


class WorldRuleSimple(BaseModel):
    category: str
    rule_text: str


class KeyLocationSimple(BaseModel):
    name: str
    description: str


class GlossaryTermSimple(BaseModel):
    term: str
    definition: str


class StoryBibleResponse(BaseModel):
    id: UUID
    story_id: UUID
    world_name: Optional[str]
    world_description: Optional[str]
    world_type: Optional[str]
    time_period: Optional[str]
    primary_locations: List[dict]
    magic_system: Optional[str]
    technology_level: Optional[str]
    central_themes: List[str]
    quick_facts: List[str]
    glossary: Any  # Can be dict or list
    # Additional structured fields for frontend
    key_locations: Optional[List[KeyLocationSimple]] = None
    themes: Optional[List[str]] = None
    world_rules: List[WorldRuleResponse]
    updated_at: datetime

    class Config:
        from_attributes = True


# Helper function
async def get_story_bible_or_404(db: AsyncSession, story_id: UUID) -> StoryBible:
    query = (
        select(StoryBible)
        .where(StoryBible.story_id == story_id)
        .options(selectinload(StoryBible.world_rules))
    )
    result = await db.execute(query)
    bible = result.scalar_one_or_none()
    if not bible:
        raise HTTPException(status_code=404, detail="Story bible not found")
    return bible


async def get_or_create_story_bible(db: AsyncSession, story_id: UUID) -> StoryBible:
    """Get story bible or create one if it doesn't exist"""
    query = (
        select(StoryBible)
        .where(StoryBible.story_id == story_id)
        .options(selectinload(StoryBible.world_rules))
    )
    result = await db.execute(query)
    bible = result.scalar_one_or_none()
    
    if not bible:
        # Create a new story bible
        bible = StoryBible(story_id=story_id)
        db.add(bible)
        await db.flush()
        await db.refresh(bible)
    
    return bible


def build_story_bible_response(bible: StoryBible) -> dict:
    """Build a response dict from the StoryBible model"""
    # Transform world_rules from DB model to simple format for frontend
    world_rules_simple = []
    if bible.world_rules:
        for rule in bible.world_rules:
            world_rules_simple.append({
                "category": rule.category.value if hasattr(rule.category, 'value') else str(rule.category),
                "rule_text": rule.description
            })
    
    # Transform primary_locations to key_locations format
    key_locations_simple = []
    if bible.primary_locations:
        for loc in bible.primary_locations:
            if isinstance(loc, dict):
                key_locations_simple.append({
                    "name": loc.get("name", "Unknown"),
                    "description": loc.get("description", "")
                })
    
    # Transform glossary to list of term/definition format
    glossary_simple = []
    if bible.glossary:
        if isinstance(bible.glossary, list):
            for item in bible.glossary:
                if isinstance(item, dict):
                    glossary_simple.append({
                        "term": item.get("term", ""),
                        "definition": item.get("definition", "")
                    })
        elif isinstance(bible.glossary, dict):
            for term, definition in bible.glossary.items():
                glossary_simple.append({
                    "term": term,
                    "definition": definition if isinstance(definition, str) else str(definition)
                })
    
    return {
        "id": bible.id,
        "story_id": bible.story_id,
        "world_name": bible.world_name,
        "world_description": bible.world_description,
        "world_type": bible.world_type,
        "time_period": bible.time_period,
        "primary_locations": bible.primary_locations or [],
        "magic_system": bible.magic_system,
        "technology_level": bible.technology_level,
        "central_themes": bible.central_themes or [],
        "quick_facts": bible.quick_facts or [],
        "glossary": glossary_simple,  # Use transformed list format for frontend
        "world_rules": world_rules_simple,
        "key_locations": key_locations_simple,
        "themes": bible.central_themes or [],
        "updated_at": bible.updated_at
    }


# Routes
@router.get("/story/{story_id}")
async def get_story_bible(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the story bible for a story (creates one if missing)"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    token_limits = await get_user_token_limits(db, current_user.id)
    token_limits = await get_user_token_limits(db, current_user.id)
    
    bible = await get_or_create_story_bible(db, story_id)
    await db.commit()
    return build_story_bible_response(bible)


@router.patch("/story/{story_id}")
async def update_story_bible(
    story_id: UUID,
    updates: StoryBibleUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update the story bible"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    bible = await get_or_create_story_bible(db, story_id)
    
    update_data = updates.model_dump(exclude_unset=True)
    
    # Handle special fields that need transformation
    
    # Handle world_rules: Convert from frontend format to WorldRule objects
    if "world_rules" in update_data and update_data["world_rules"] is not None:
        # Delete existing rules first
        for rule in list(bible.world_rules or []):
            await db.delete(rule)
        
        # Create new rules from the frontend format
        for rule_data in update_data["world_rules"]:
            category_map = {
                "physics": RuleCategory.PHYSICS,
                "magic": RuleCategory.MAGIC,
                "society": RuleCategory.SOCIETY,
                "technology": RuleCategory.TECHNOLOGY,
                "biology": RuleCategory.BIOLOGY,
                "geography": RuleCategory.GEOGRAPHY,
                "history": RuleCategory.HISTORY,
            }
            category_str = rule_data.get("category", "custom").lower()
            category = category_map.get(category_str, RuleCategory.CUSTOM)
            
            rule = WorldRule(
                story_bible_id=bible.id,
                title=rule_data.get("category", "Rule"),  # Use category as title
                description=rule_data.get("rule_text", ""),
                category=category,
                importance=5,
                is_strict=True
            )
            db.add(rule)
        del update_data["world_rules"]
    
    # Handle key_locations: Convert to primary_locations format
    if "key_locations" in update_data and update_data["key_locations"] is not None:
        # Transform key_locations to primary_locations format
        bible.primary_locations = [
            {"name": loc.get("name", ""), "description": loc.get("description", ""), "importance": "medium"}
            for loc in update_data["key_locations"]
        ]
        del update_data["key_locations"]
    
    # Handle glossary: Convert from list of {term, definition} to proper format
    if "glossary" in update_data and update_data["glossary"] is not None:
        glossary = update_data["glossary"]
        if isinstance(glossary, list):
            # Keep as list format (works with frontend)
            bible.glossary = glossary
        else:
            # Already a dict, keep as is
            bible.glossary = glossary
        del update_data["glossary"]
    
    # Handle themes: Map to central_themes
    if "themes" in update_data and update_data["themes"] is not None:
        bible.central_themes = update_data["themes"]
        del update_data["themes"]
    
    # Apply remaining updates
    for key, value in update_data.items():
        if hasattr(bible, key):
            setattr(bible, key, value)
    
    bible.updated_at = datetime.utcnow()
    await db.flush()
    
    # Refresh to get rules
    await db.refresh(bible)
    
    # Embed story bible for semantic search (RAG)
    try:
        embedded_count = await memory_service.embed_story_bible(
            db=db,
            story_id=str(story_id),
            story_bible=bible
        )
        logger.info(f"✓ Embedded story bible with {embedded_count} entries")
    except Exception as e:
        logger.warning(f"Failed to embed story bible: {e}")
    
    await db.commit()
    
    return build_story_bible_response(bible)


@router.post("/story/{story_id}/rules", response_model=WorldRuleResponse, status_code=status.HTTP_201_CREATED)
async def create_world_rule(
    story_id: UUID,
    rule_data: WorldRuleCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a world rule to the story bible"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    bible = await get_or_create_story_bible(db, story_id)
    
    rule = WorldRule(
        story_bible_id=bible.id,
        **rule_data.model_dump()
    )
    
    db.add(rule)
    await db.flush()
    
    return WorldRuleResponse.model_validate(rule)


@router.get("/story/{story_id}/rules", response_model=List[WorldRuleResponse])
async def list_world_rules(
    story_id: UUID,
    category: Optional[RuleCategory] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all world rules for a story"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    bible = await get_or_create_story_bible(db, story_id)
    
    rules = bible.world_rules
    if category:
        rules = [r for r in rules if r.category == category]
    
    return [WorldRuleResponse.model_validate(r) for r in rules]


@router.patch("/rules/{rule_id}", response_model=WorldRuleResponse)
async def update_world_rule(
    rule_id: UUID,
    updates: WorldRuleUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a world rule"""
    query = select(WorldRule).where(WorldRule.id == rule_id)
    result = await db.execute(query)
    rule = result.scalar_one_or_none()
    
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    # Verify ownership through story bible
    bible_query = select(StoryBible).where(StoryBible.id == rule.story_bible_id)
    bible_result = await db.execute(bible_query)
    bible = bible_result.scalar_one_or_none()
    
    story = await story_service.get_story(db, bible.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_data = updates.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(rule, key, value)
    
    rule.updated_at = datetime.utcnow()
    await db.flush()
    
    return WorldRuleResponse.model_validate(rule)


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_world_rule(
    rule_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a world rule"""
    query = select(WorldRule).where(WorldRule.id == rule_id)
    result = await db.execute(query)
    rule = result.scalar_one_or_none()
    
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    # Verify ownership
    bible_query = select(StoryBible).where(StoryBible.id == rule.story_bible_id)
    bible_result = await db.execute(bible_query)
    bible = bible_result.scalar_one_or_none()
    
    story = await story_service.get_story(db, bible.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.delete(rule)


@router.post("/story/{story_id}/locations")
async def add_location(
    story_id: UUID,
    name: str,
    description: str,
    importance: int = 5,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a location to the story bible"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    bible = await get_or_create_story_bible(db, story_id)
    
    locations = bible.primary_locations or []
    locations.append({
        "name": name,
        "description": description,
        "importance": importance
    })
    bible.primary_locations = locations
    bible.updated_at = datetime.utcnow()
    
    await db.flush()
    
    return {"message": "Location added", "locations": locations}


@router.post("/story/{story_id}/glossary")
async def add_glossary_term(
    story_id: UUID,
    term: str,
    definition: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a term to the glossary"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    bible = await get_or_create_story_bible(db, story_id)
    
    glossary = bible.glossary or {}
    glossary[term] = definition
    bible.glossary = glossary
    bible.updated_at = datetime.utcnow()
    
    await db.flush()
    
    return {"message": "Term added", "glossary": glossary}


@router.post("/story/{story_id}/generate")
async def auto_generate_story_bible(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Auto-generate Story Bible by analyzing all story content.
    Uses AI to extract world rules, locations, terminology, themes, etc.
    """
    try:
        return await _do_generate_story_bible(story_id, current_user, db)
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[STORY BIBLE ERROR]\n{tb}\n", flush=True)
        logger.error(f"Story Bible generate unhandled error:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc!r}")


async def _do_generate_story_bible(
    story_id: UUID,
    current_user: User,
    db: AsyncSession
):
    """Inner implementation of story bible generation."""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Get all chapters content
    chapters = await chapter_service.get_chapters_by_story(db, story_id)
    if not chapters:
        raise HTTPException(
            status_code=400, 
            detail="Cannot generate Story Bible without any chapter content"
        )
    
    # Combine all chapter content
    all_content = ""
    for chapter in chapters:
        if chapter.content:
            all_content += f"\n\n--- Chapter {chapter.number}: {chapter.title} ---\n\n"
            all_content += chapter.content
    
    if len(all_content.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Not enough story content to generate Story Bible. Write more first!"
        )
    
    # Get existing characters for context
    characters = await character_service.get_characters_by_story(db, story_id)
    characters_str = ""
    if characters:
        characters_str = ", ".join([f"{c.name} ({c.role.value})" for c in characters])
    
    # Limit content for API — CPU Ollama runs ~9 tok/s; keep prompt under 200 tokens.
    # 600 chars ≈ 150 tokens, leaving headroom for schema + output (200 tokens).
    if len(all_content) > 600:
        all_content = all_content[:600]
    
    logger.info(f"Generating Story Bible for story {story_id} from {len(all_content)} chars")
    token_limits = await get_user_token_limits(db, current_user.id)

    def enum_val(field, default):
        """Safely get enum value whether field is an Enum, str, or None."""
        if field is None:
            return default
        if hasattr(field, 'value'):
            return field.value
        return str(field).lower()

    # Generate Story Bible using AI - use simple fast prompt first
    result = await gemini_service.generate_story_bible_simple(
        story_content=all_content,
        story_title=story.title,
        story_genre=enum_val(story.genre, "general"),
        max_tokens=token_limits["max_tokens_story_bible"]
    )

    # If parse failed, try the full prompt with even shorter content (300 chars)
    if not result.get("parsed") or not result.get("bible_data"):
        logger.warning(f"Simple Story Bible parse failed, trying full prompt for story {story_id}")
        short_content = all_content[:300]  # Keep tiny for CPU inference
        result = await gemini_service.generate_story_bible(
            story_content=short_content,
            story_title=story.title,
            story_genre=enum_val(story.genre, "general"),
            story_tone=enum_val(story.tone, "neutral"),
            existing_characters=characters_str,
            language=story.language or "English",
            max_tokens=token_limits["max_tokens_story_bible"]
        )

    if not result.get("success"):
        error_msg = result.get('error', 'Unknown error')
        if 'timeout' in error_msg.lower() or 'ReadTimeout' in error_msg:
            raise HTTPException(
                status_code=504,
                detail="AI took too long to generate Story Bible. Try with less content or try again later."
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Story Bible: {error_msg}"
        )

    # If still no valid JSON, try harder before falling back to minimal bible
    if not result.get("parsed") or not result.get("bible_data"):
        raw = result.get("content", "")
        # Last-chance: attempt to parse the raw content directly
        # (handles case where simple prompt succeeded but parser missed it)
        last_chance = gemini_service._parse_json_from_text(raw) if raw else None
        if last_chance and isinstance(last_chance, dict):
            logger.info(f"Story Bible last-chance JSON parse succeeded for story {story_id}")
            result["bible_data"] = last_chance
            result["parsed"] = True
        else:
            logger.warning(f"Story Bible JSON parse failed for story {story_id}. Building minimal bible.")
            # Don't put raw JSON text into world_description
            bible_data = {
                "world_description": "Generated from story content. Edit to add details.",
                "world_type": enum_val(story.genre, "general"),
                "central_themes": [],
                "quick_facts": [],
                "glossary": [],
                "world_rules": [],
                "primary_locations": [],
            }

    if result.get("parsed") and result.get("bible_data"):
        bible_data = result["bible_data"]
    # else: bible_data was already set by the minimal fallback block above
    
    # Get or create the story bible
    bible = await get_or_create_story_bible(db, story_id)
    
    # Update with generated data
    if bible_data.get("world_name"):
        bible.world_name = bible_data["world_name"]
    if bible_data.get("world_description"):
        bible.world_description = bible_data["world_description"]
    if bible_data.get("world_type"):
        bible.world_type = bible_data["world_type"]
    if bible_data.get("time_period"):
        bible.time_period = bible_data["time_period"]
    if bible_data.get("primary_locations"):
        bible.primary_locations = bible_data["primary_locations"]
    if bible_data.get("magic_system"):
        bible.magic_system = bible_data["magic_system"]
    if bible_data.get("magic_rules"):
        bible.magic_rules = bible_data["magic_rules"]
    if bible_data.get("magic_limitations"):
        bible.magic_limitations = bible_data["magic_limitations"]
    if bible_data.get("technology_level"):
        bible.technology_level = bible_data["technology_level"]
    if bible_data.get("societies"):
        bible.societies = bible_data["societies"]
    if bible_data.get("central_themes"):
        bible.central_themes = bible_data["central_themes"]
    if bible_data.get("recurring_motifs"):
        bible.recurring_motifs = bible_data["recurring_motifs"]
    if bible_data.get("tone_guidelines"):
        bible.tone_guidelines = bible_data["tone_guidelines"]
    if bible_data.get("quick_facts"):
        bible.quick_facts = bible_data["quick_facts"]
    
    # Handle glossary (convert from list to dict if needed)
    if bible_data.get("glossary"):
        glossary = bible_data["glossary"]
        if isinstance(glossary, list):
            bible.glossary = glossary  # Keep as list format
        else:
            bible.glossary = glossary
    
    # Create world rules from generated data
    if bible_data.get("world_rules"):
        for rule_data in bible_data["world_rules"]:
            # Map category string to enum
            category_map = {
                "physics": RuleCategory.PHYSICS,
                "magic": RuleCategory.MAGIC,
                "society": RuleCategory.SOCIETY,
                "technology": RuleCategory.TECHNOLOGY,
                "biology": RuleCategory.BIOLOGY,
                "geography": RuleCategory.GEOGRAPHY,
                "history": RuleCategory.HISTORY,
            }
            category = category_map.get(
                rule_data.get("category", "custom").lower(),
                RuleCategory.CUSTOM
            )
            
            rule = WorldRule(
                story_bible_id=bible.id,
                title=rule_data.get("title", "Untitled Rule"),
                description=rule_data.get("description", ""),
                category=category,
                importance=rule_data.get("importance", 5),
                is_strict=True
            )
            db.add(rule)
    
    bible.updated_at = datetime.utcnow()
    await db.flush()
    await db.commit()

    # Reload bible with world_rules eagerly loaded (db.refresh doesn't load relationships)
    result_q = await db.execute(
        select(StoryBible)
        .where(StoryBible.id == bible.id)
        .options(selectinload(StoryBible.world_rules))
    )
    bible = result_q.scalar_one()

    # Embed the story bible for semantic search
    try:
        embedded_count = await memory_service.embed_story_bible(
            db=db,
            story_id=str(story_id),
            story_bible=bible
        )
        logger.info(f"✓ Auto-generated and embedded Story Bible with {embedded_count} entries")
    except Exception as e:
        logger.warning(f"Failed to embed story bible: {e}")

    return build_story_bible_response(bible)


@router.post("/story/{story_id}/update-from-content")
async def update_story_bible_from_new_content(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Incrementally update Story Bible based on new content.
    Analyzes recent chapters and adds new elements without removing existing ones.
    """
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Get existing bible
    bible = await get_or_create_story_bible(db, story_id)
    
    # Get recent chapter content (last 2 chapters)
    chapters = await chapter_service.get_chapters_by_story(db, story_id)
    recent_content = ""
    for chapter in chapters[-2:]:
        if chapter.content:
            recent_content += f"\n\n{chapter.content}"
    
    if len(recent_content.strip()) < 50:
        return {"message": "Not enough new content to analyze", "updated": False}
    
    # Build existing bible dict for comparison
    existing_bible = {
        "locations": bible.primary_locations or [],
        "world_rules": [{"title": r.title, "description": r.description} for r in (bible.world_rules or [])],
        "glossary": bible.glossary or [],
        "themes": bible.central_themes or [],
        "quick_facts": bible.quick_facts or []
    }

    token_limits = await get_user_token_limits(db, current_user.id)

    # Get updates from AI
    result = await gemini_service.update_story_bible_from_content(
        new_content=recent_content[:8000],
        existing_bible=existing_bible,
        story_genre=story.genre.value if story.genre else "general",
        language=story.language or "English",
        max_tokens=token_limits["max_tokens_story_bible_update"]
    )
    
    if not result.get("success") or not result.get("parsed"):
        return {"message": "Could not extract new elements", "updated": False}
    
    updates = result.get("updates", {})
    added_items = []
    
    # Add new locations
    if updates.get("new_locations"):
        for loc in updates["new_locations"]:
            if loc.get("name"):
                bible.primary_locations = (bible.primary_locations or []) + [loc]
                added_items.append(f"Location: {loc['name']}")
    
    # Add new world rules
    if updates.get("new_world_rules"):
        for rule_data in updates["new_world_rules"]:
            if rule_data.get("title"):
                category_map = {
                    "physics": RuleCategory.PHYSICS,
                    "magic": RuleCategory.MAGIC,
                    "society": RuleCategory.SOCIETY,
                    "technology": RuleCategory.TECHNOLOGY,
                }
                category = category_map.get(
                    rule_data.get("category", "custom").lower(),
                    RuleCategory.CUSTOM
                )
                rule = WorldRule(
                    story_bible_id=bible.id,
                    title=rule_data["title"],
                    description=rule_data.get("description", ""),
                    category=category,
                    importance=rule_data.get("importance", 5)
                )
                db.add(rule)
                added_items.append(f"Rule: {rule_data['title']}")
    
    # Add new glossary terms
    if updates.get("new_glossary_terms"):
        glossary = bible.glossary or []
        if isinstance(glossary, dict):
            glossary = [{"term": k, "definition": v} for k, v in glossary.items()]
        for term_data in updates["new_glossary_terms"]:
            if term_data.get("term"):
                glossary.append(term_data)
                added_items.append(f"Term: {term_data['term']}")
        bible.glossary = glossary
    
    # Add new themes
    if updates.get("new_themes"):
        bible.central_themes = (bible.central_themes or []) + updates["new_themes"]
        for theme in updates["new_themes"]:
            added_items.append(f"Theme: {theme}")
    
    # Add new quick facts
    if updates.get("new_quick_facts"):
        bible.quick_facts = (bible.quick_facts or []) + updates["new_quick_facts"]
        for fact in updates["new_quick_facts"]:
            added_items.append(f"Fact: {fact[:50]}...")
    
    if added_items:
        bible.updated_at = datetime.utcnow()
        await db.flush()
        
        # Re-embed story bible
        try:
            await memory_service.embed_story_bible(db, str(story_id), bible)
        except Exception as e:
            logger.warning(f"Failed to re-embed story bible: {e}")
        
        await db.commit()
        
        return {
            "message": f"Added {len(added_items)} new elements to Story Bible",
            "updated": True,
            "added_items": added_items
        }
    
    return {"message": "No new elements found to add", "updated": False}

