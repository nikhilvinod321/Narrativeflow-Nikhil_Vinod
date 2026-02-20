"""
Chapters Routes - CRUD operations for chapters
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import Optional, List
from uuid import UUID
import asyncio
import logging

from app.database import get_db, get_async_session
from app.models.user import User
from app.models.chapter import Chapter, ChapterStatus
from app.services.chapter_service import ChapterService
from app.services.story_service import StoryService
from app.services.gemini_service import GeminiService
from app.services.memory_service import MemoryService
from app.services.character_service import CharacterService
from app.services.token_settings import get_user_token_limits
from app.routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()
chapter_service = ChapterService()
story_service = StoryService()
gemini_service = GeminiService()
memory_service = MemoryService()
character_service = CharacterService()


# Pydantic models
class ChapterCreate(BaseModel):
    story_id: UUID
    title: str
    content: str = ""
    outline: Optional[str] = None
    notes: Optional[str] = None


class ChapterUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    outline: Optional[str] = None
    notes: Optional[str] = None
    summary: Optional[str] = None
    status: Optional[ChapterStatus] = None
    is_locked: Optional[bool] = None
    pov_character_id: Optional[UUID] = None
    target_word_count: Optional[int] = None
    story_timeline_start: Optional[str] = None
    story_timeline_end: Optional[str] = None


class ChapterResponse(BaseModel):
    id: UUID
    story_id: UUID
    title: str
    number: int
    order: int
    content: str
    summary: Optional[str]
    notes: Optional[str]
    outline: Optional[str]
    status: ChapterStatus
    is_locked: bool
    word_count: int
    target_word_count: Optional[int]
    reading_time_minutes: int
    pov_character_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChapterListResponse(BaseModel):
    id: UUID
    title: str
    number: int
    order: int
    content: str  # Include content so editor can display it
    status: ChapterStatus
    word_count: int
    summary: Optional[str]
    updated_at: datetime

    class Config:
        from_attributes = True


class ContentUpdate(BaseModel):
    content: str
    append: bool = False


class ReorderRequest(BaseModel):
    chapter_order: List[UUID]


# Background task for automatic Story Bible updates
async def trigger_story_bible_update(story_id: UUID):
    """
    Background task to update Story Bible when chapter content changes.
    Uses a new database session to avoid conflicts.
    """
    try:
        logger.info(f"🔄 Triggering automatic Story Bible update for story {story_id}")
        
        # Import here to avoid circular imports
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        from app.models.story_bible import StoryBible, WorldRule, RuleCategory
        
        async with get_async_session() as db:
            # Get the story
            story = await story_service.get_story(db, story_id)
            if not story:
                logger.warning(f"Story {story_id} not found for bible update")
                return
            token_limits = await get_user_token_limits(db, story.author_id)
            
            # Get all chapters
            chapters = await chapter_service.get_chapters_by_story(db, story_id)
            if not chapters:
                return
            
            # Calculate total content
            total_content = ""
            for chapter in chapters:
                if chapter.content:
                    total_content += chapter.content + " "
            
            # Need at least 100 chars to do anything meaningful
            if len(total_content.strip()) < 100:
                return
            
            # Check if Story Bible exists
            query = (
                select(StoryBible)
                .where(StoryBible.story_id == story_id)
                .options(selectinload(StoryBible.world_rules))
            )
            result = await db.execute(query)
            bible = result.scalar_one_or_none()
            
            # If no bible exists or it's empty, generate a new one
            if not bible or (not bible.world_name and not bible.primary_locations and not bible.central_themes):
                logger.info(f"📖 Auto-generating initial Story Bible for story {story_id}")
                
                # Create bible if needed
                if not bible:
                    bible = StoryBible(story_id=story_id)
                    db.add(bible)
                    await db.flush()
                
                # Get characters for context
                characters = await character_service.get_characters_by_story(db, story_id)
                characters_str = ""
                if characters:
                    characters_str = ", ".join([f"{c.name} ({c.role.value})" for c in characters])
                
                # Limit content
                content_for_gen = total_content[:8000] if len(total_content) > 8000 else total_content
                
                # Generate Story Bible
                result = await gemini_service.generate_story_bible(
                    story_content=content_for_gen,
                    story_title=story.title,
                    story_genre=story.genre.value if story.genre else "general",
                    story_tone=story.tone.value if story.tone else "neutral",
                    existing_characters=characters_str,
                    language=story.language or "English",
                    max_tokens=token_limits["max_tokens_story_bible"]
                )
                
                if result.get("success") and result.get("parsed") and result.get("bible_data"):
                    bible_data = result["bible_data"]
                    
                    # Update bible with generated data
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
                    if bible_data.get("technology_level"):
                        bible.technology_level = bible_data["technology_level"]
                    if bible_data.get("central_themes"):
                        bible.central_themes = bible_data["central_themes"]
                    if bible_data.get("quick_facts"):
                        bible.quick_facts = bible_data["quick_facts"]
                    if bible_data.get("glossary"):
                        bible.glossary = bible_data["glossary"]
                    
                    # Create world rules
                    if bible_data.get("world_rules"):
                        for rule_data in bible_data["world_rules"]:
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
                                title=rule_data.get("title", "Rule"),
                                description=rule_data.get("description", ""),
                                category=category,
                                importance=rule_data.get("importance", 5)
                            )
                            db.add(rule)
                    
                    bible.updated_at = datetime.utcnow()
                    logger.info(f"✅ Auto-generated Story Bible for story {story_id}")
                else:
                    logger.warning(f"Failed to auto-generate Story Bible: {result.get('error', 'Unknown')}")
            else:
                # Bible exists, do incremental update
                logger.info(f"📝 Incrementally updating Story Bible for story {story_id}")
                
                # Get recent content (last 2 chapters)
                recent_content = ""
                for chapter in chapters[-2:]:
                    if chapter.content:
                        recent_content += f"\n\n{chapter.content}"
                
                if len(recent_content.strip()) >= 50:
                    # Build existing bible for comparison
                    existing_bible = {
                        "locations": bible.primary_locations or [],
                        "world_rules": [{"title": r.title, "description": r.description} for r in (bible.world_rules or [])],
                        "glossary": bible.glossary or [],
                        "themes": bible.central_themes or [],
                        "quick_facts": bible.quick_facts or []
                    }
                    
                    # Get updates from AI
                    result = await gemini_service.update_story_bible_from_content(
                        new_content=recent_content[:8000],
                        existing_bible=existing_bible,
                        story_genre=story.genre.value if story.genre else "general",
                        language=story.language or "English",
                        max_tokens=token_limits["max_tokens_story_bible_update"]
                    )
                    
                    if result.get("success") and result.get("parsed"):
                        updates = result.get("updates", {})
                        added_count = 0
                        
                        # Add new locations
                        if updates.get("new_locations"):
                            for loc in updates["new_locations"]:
                                if loc.get("name"):
                                    bible.primary_locations = (bible.primary_locations or []) + [loc]
                                    added_count += 1
                        
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
                                    added_count += 1
                        
                        # Add new glossary terms
                        if updates.get("new_glossary_terms"):
                            glossary = bible.glossary or []
                            if isinstance(glossary, dict):
                                glossary = [{"term": k, "definition": v} for k, v in glossary.items()]
                            for term_data in updates["new_glossary_terms"]:
                                if term_data.get("term"):
                                    glossary.append(term_data)
                                    added_count += 1
                            bible.glossary = glossary
                        
                        # Add new themes
                        if updates.get("new_themes"):
                            bible.central_themes = (bible.central_themes or []) + updates["new_themes"]
                            added_count += len(updates["new_themes"])
                        
                        if added_count > 0:
                            bible.updated_at = datetime.utcnow()
                            logger.info(f"✅ Added {added_count} new elements to Story Bible")
            
            # Re-embed story bible for RAG
            try:
                await db.refresh(bible)
                embedded_count = await memory_service.embed_story_bible(db, str(story_id), bible)
                logger.info(f"✓ Re-embedded Story Bible: {embedded_count} entries")
            except Exception as e:
                logger.warning(f"Failed to embed story bible: {e}")
            
            await db.commit()
            
    except Exception as e:
        logger.error(f"Error in automatic Story Bible update: {e}")


# Debounce mechanism to avoid too many updates
_story_bible_update_tasks = {}

def schedule_story_bible_update(story_id: UUID, delay_seconds: int = 30):
    """
    Schedule a Story Bible update with debouncing.
    If another update is scheduled within the delay, it will be cancelled.
    """
    task_key = str(story_id)
    
    # Cancel any existing scheduled task
    if task_key in _story_bible_update_tasks:
        _story_bible_update_tasks[task_key].cancel()
        del _story_bible_update_tasks[task_key]
    
    async def delayed_update():
        try:
            await asyncio.sleep(delay_seconds)
            await trigger_story_bible_update(story_id)
        except asyncio.CancelledError:
            pass
        finally:
            if task_key in _story_bible_update_tasks:
                del _story_bible_update_tasks[task_key]
    
    # Schedule new task
    task = asyncio.create_task(delayed_update())
    _story_bible_update_tasks[task_key] = task
    logger.info(f"📅 Scheduled Story Bible update for story {story_id} in {delay_seconds}s")


# Routes
@router.post("", response_model=ChapterResponse, status_code=status.HTTP_201_CREATED)
async def create_chapter(
    chapter_data: ChapterCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new chapter"""
    # Verify story ownership
    story = await story_service.get_story(db, chapter_data.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    chapter = await chapter_service.create_chapter(
        db=db,
        story_id=chapter_data.story_id,
        title=chapter_data.title,
        content=chapter_data.content,
        outline=chapter_data.outline,
        notes=chapter_data.notes
    )
    
    return ChapterResponse.model_validate(chapter)


@router.get("/story/{story_id}", response_model=List[ChapterListResponse])
async def list_chapters(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all chapters for a story"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    chapters = await chapter_service.get_chapters_by_story(db, story_id)
    return [ChapterListResponse.model_validate(c) for c in chapters]


@router.get("/{chapter_id}", response_model=ChapterResponse)
async def get_chapter(
    chapter_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific chapter"""
    chapter = await chapter_service.get_chapter(db, chapter_id)
    
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, chapter.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return ChapterResponse.model_validate(chapter)


@router.patch("/{chapter_id}", response_model=ChapterResponse)
async def update_chapter(
    chapter_id: UUID,
    updates: ChapterUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a chapter"""
    chapter = await chapter_service.get_chapter(db, chapter_id)
    
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, chapter.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Check if locked
    if chapter.is_locked and not updates.is_locked:
        raise HTTPException(status_code=400, detail="Chapter is locked")
    
    updated = await chapter_service.update_chapter(
        db, chapter_id, updates.model_dump(exclude_unset=True)
    )
    
    # Update story word count
    await story_service.update_word_count(db, story.id)
    
    # Auto-embed chapter content if content was changed
    if updates.content is not None and len(updates.content) > 100:
        try:
            characters = await character_service.get_characters_by_story(db, story.id)
            character_names = [c.name for c in characters] if characters else []
            await memory_service.embed_chapter(
                db=db,
                story_id=str(story.id),
                chapter_id=str(chapter_id),
                content=updates.content,
                chapter_metadata={
                    "title": updated.title,
                    "number": updated.number,
                    "characters": character_names
                }
            )
            logger.info(f"✓ Auto-embedded chapter {chapter_id}")
        except Exception as e:
            logger.warning(f"Failed to auto-embed chapter: {e}")
        
        # Schedule Story Bible update
        schedule_story_bible_update(story.id, delay_seconds=60)
    
    return ChapterResponse.model_validate(updated)


@router.put("/{chapter_id}/content", response_model=ChapterResponse)
async def update_chapter_content(
    chapter_id: UUID,
    content_update: ContentUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update chapter content specifically (for auto-save)"""
    chapter = await chapter_service.get_chapter(db, chapter_id)
    
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, chapter.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if chapter.is_locked:
        raise HTTPException(status_code=400, detail="Chapter is locked")
    
    updated = await chapter_service.update_chapter_content(
        db, chapter_id, content_update.content, content_update.append
    )
    
    # Update story word count
    await story_service.update_word_count(db, story.id)
    
    # Auto-embed chapter content
    if len(content_update.content) > 100:
        try:
            characters = await character_service.get_characters_by_story(db, story.id)
            character_names = [c.name for c in characters] if characters else []
            await memory_service.embed_chapter(
                db=db,
                story_id=str(story.id),
                chapter_id=str(chapter_id),
                content=updated.content,
                chapter_metadata={
                    "title": updated.title,
                    "number": updated.number,
                    "characters": character_names
                }
            )
            logger.info(f"✓ Auto-embedded chapter {chapter_id} on content save")
        except Exception as e:
            logger.warning(f"Failed to auto-embed chapter: {e}")
    
    # Schedule Story Bible update if there's substantial content
    if len(content_update.content) > 200:
        schedule_story_bible_update(story.id, delay_seconds=120)
    
    return ChapterResponse.model_validate(updated)


@router.delete("/{chapter_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chapter(
    chapter_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a chapter"""
    chapter = await chapter_service.get_chapter(db, chapter_id)
    
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, chapter.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await chapter_service.delete_chapter(db, chapter_id)


@router.post("/story/{story_id}/reorder", response_model=List[ChapterListResponse])
async def reorder_chapters(
    story_id: UUID,
    reorder: ReorderRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Reorder chapters in a story"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    chapters = await chapter_service.reorder_chapters(db, story_id, reorder.chapter_order)
    return [ChapterListResponse.model_validate(c) for c in chapters]


@router.get("/{chapter_id}/context")
async def get_chapter_context(
    chapter_id: UUID,
    include_previous: int = 1,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get chapter with surrounding context for AI generation"""
    chapter = await chapter_service.get_chapter(db, chapter_id)
    
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, chapter.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    context = await chapter_service.get_chapter_context(db, chapter_id, include_previous)
    
    return {
        "chapter": ChapterResponse.model_validate(context["chapter"]),
        "previous_chapters": [
            ChapterListResponse.model_validate(c) for c in context["previous_chapters"]
        ]
    }
