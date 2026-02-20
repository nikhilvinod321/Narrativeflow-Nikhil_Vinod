"""
Stories Routes - CRUD operations for stories
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from app.database import get_db
from app.models.user import User
from app.models.story import Story, StoryGenre, StoryTone, StoryStatus
from app.services.story_service import StoryService
from app.routes.auth import get_current_user

router = APIRouter()
story_service = StoryService()


# Pydantic models
class StoryCreate(BaseModel):
    title: str
    genre: StoryGenre = StoryGenre.OTHER
    tone: StoryTone = StoryTone.SERIOUS
    language: str = "English"
    logline: Optional[str] = None
    synopsis: Optional[str] = None
    pov_style: str = "third_person_limited"
    tense: str = "past"
    setting_time: Optional[str] = None
    setting_place: Optional[str] = None
    writing_style: Optional[str] = None


class StoryUpdate(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    logline: Optional[str] = None
    synopsis: Optional[str] = None
    genre: Optional[StoryGenre] = None
    secondary_genre: Optional[StoryGenre] = None
    tone: Optional[StoryTone] = None
    status: Optional[StoryStatus] = None
    setting_time: Optional[str] = None
    setting_place: Optional[str] = None
    pov_style: Optional[str] = None
    tense: Optional[str] = None
    language: Optional[str] = None
    writing_style: Optional[str] = None
    ai_personality: Optional[str] = None
    creativity_level: Optional[int] = None
    cover_image_url: Optional[str] = None
    tags: Optional[List[str]] = None


class StoryResponse(BaseModel):
    id: UUID
    title: str
    subtitle: Optional[str] = None
    logline: Optional[str] = None
    synopsis: Optional[str] = None
    genre: StoryGenre
    secondary_genre: Optional[StoryGenre] = None
    tone: StoryTone
    status: StoryStatus
    pov_style: str
    tense: str
    language: str = "English"
    setting_time: Optional[str] = None
    setting_place: Optional[str] = None
    word_count: int = 0
    chapter_count: int = 0
    cover_image_url: Optional[str] = None
    tags: List[str] = []
    created_at: datetime
    updated_at: datetime
    last_written_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class StoryListResponse(BaseModel):
    id: UUID
    title: str
    genre: StoryGenre
    tone: StoryTone
    status: StoryStatus
    word_count: int
    chapter_count: int
    cover_image_url: Optional[str]
    updated_at: datetime

    class Config:
        from_attributes = True


class StoryStatsResponse(BaseModel):
    word_count: int
    chapter_count: int
    character_count: int
    plotline_count: int
    status: str
    created_at: datetime
    updated_at: datetime
    last_written_at: Optional[datetime]


# Routes
@router.post("", response_model=StoryResponse, status_code=status.HTTP_201_CREATED)
async def create_story(
    story_data: StoryCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new story"""
    story = await story_service.create_story(
        db=db,
        author_id=current_user.id,
        **story_data.model_dump()
    )
    return StoryResponse.model_validate(story)


@router.get("", response_model=List[StoryListResponse])
async def list_stories(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    status_filter: Optional[StoryStatus] = Query(None),
    genre_filter: Optional[StoryGenre] = Query(None)
):
    """List all stories for the current user"""
    stories = await story_service.get_stories_by_author(db, current_user.id)
    
    # Apply filters
    if status_filter:
        stories = [s for s in stories if s.status == status_filter]
    if genre_filter:
        stories = [s for s in stories if s.genre == genre_filter]
    
    return [StoryListResponse.model_validate(s) for s in stories]


@router.get("/{story_id}", response_model=StoryResponse)
async def get_story(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific story"""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Fetching story {story_id} for user {current_user.id} ({current_user.email})")
    
    story = await story_service.get_story(db, story_id)
    
    if not story:
        logger.warning(f"Story {story_id} not found in database")
        raise HTTPException(status_code=404, detail="Story not found")
    
    logger.info(f"Story found: author_id={story.author_id}, current_user.id={current_user.id}")
    
    if story.author_id != current_user.id:
        logger.warning(f"User {current_user.id} not authorized to access story {story_id} (owner: {story.author_id})")
        raise HTTPException(status_code=403, detail="Not authorized to access this story")
    
    return StoryResponse.model_validate(story)


@router.patch("/{story_id}", response_model=StoryResponse)
async def update_story(
    story_id: UUID,
    updates: StoryUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a story"""
    story = await story_service.get_story(db, story_id)
    
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    if story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this story")
    
    updated = await story_service.update_story(
        db, story_id, updates.model_dump(exclude_unset=True)
    )
    
    return StoryResponse.model_validate(updated)


@router.delete("/{story_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_story(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a story"""
    story = await story_service.get_story(db, story_id)
    
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    if story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this story")
    
    await story_service.delete_story(db, story_id)


@router.get("/{story_id}/stats", response_model=StoryStatsResponse)
async def get_story_stats(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get story statistics"""
    story = await story_service.get_story(db, story_id)
    
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    if story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this story")
    
    stats = await story_service.get_story_stats(db, story_id)
    return StoryStatsResponse(**stats)


@router.post("/{story_id}/archive")
async def archive_story(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Archive a story"""
    story = await story_service.get_story(db, story_id)
    
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    await story_service.update_story(db, story_id, {"status": StoryStatus.ARCHIVED})
    
    return {"message": "Story archived successfully"}
