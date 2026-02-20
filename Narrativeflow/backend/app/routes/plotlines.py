"""
Plotlines Routes - CRUD operations for plotlines
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from app.database import get_db
from app.models.user import User
from app.models.plotline import Plotline, PlotlineType, PlotlineStatus
from app.services.story_service import StoryService
from app.routes.auth import get_current_user

router = APIRouter()
story_service = StoryService()


# Pydantic models
class PlotlineCreate(BaseModel):
    story_id: UUID
    title: str
    description: Optional[str] = None
    type: PlotlineType = PlotlineType.SUBPLOT
    importance: int = 5
    setup: Optional[str] = None
    development: Optional[str] = None
    climax: Optional[str] = None
    resolution: Optional[str] = None
    primary_characters: List[UUID] = []
    color: Optional[str] = None


class PlotlineUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[PlotlineType] = None
    status: Optional[PlotlineStatus] = None
    importance: Optional[int] = None
    setup: Optional[str] = None
    development: Optional[str] = None
    climax: Optional[str] = None
    resolution: Optional[str] = None
    primary_characters: Optional[List[UUID]] = None
    secondary_characters: Optional[List[UUID]] = None
    introduction_chapter: Optional[int] = None
    resolution_chapter: Optional[int] = None
    connected_plotlines: Optional[List[UUID]] = None
    color: Optional[str] = None


class PlotPointCreate(BaseModel):
    chapter_id: Optional[UUID] = None
    description: str
    order: int


class PlotlineResponse(BaseModel):
    id: UUID
    story_id: UUID
    title: str
    description: Optional[str]
    type: PlotlineType
    status: PlotlineStatus
    importance: int
    setup: Optional[str]
    development: Optional[str]
    climax: Optional[str]
    resolution: Optional[str]
    primary_characters: List[str]
    secondary_characters: List[str]
    introduction_chapter: Optional[int]
    resolution_chapter: Optional[int]
    plot_points: List[dict]
    open_questions: List[str]
    color: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PlotlineListResponse(BaseModel):
    id: UUID
    title: str
    type: PlotlineType
    status: PlotlineStatus
    importance: int
    color: Optional[str]

    class Config:
        from_attributes = True


# Helper function
async def get_plotline_or_404(db: AsyncSession, plotline_id: UUID) -> Plotline:
    query = select(Plotline).where(Plotline.id == plotline_id)
    result = await db.execute(query)
    plotline = result.scalar_one_or_none()
    if not plotline:
        raise HTTPException(status_code=404, detail="Plotline not found")
    return plotline


# Routes
@router.post("", response_model=PlotlineResponse, status_code=status.HTTP_201_CREATED)
async def create_plotline(
    plotline_data: PlotlineCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new plotline"""
    # Verify story ownership
    story = await story_service.get_story(db, plotline_data.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    plotline = Plotline(
        story_id=plotline_data.story_id,
        title=plotline_data.title,
        description=plotline_data.description,
        type=plotline_data.type,
        status=PlotlineStatus.PLANNED,
        importance=plotline_data.importance,
        setup=plotline_data.setup,
        development=plotline_data.development,
        climax=plotline_data.climax,
        resolution=plotline_data.resolution,
        primary_characters=[str(c) for c in plotline_data.primary_characters],
        color=plotline_data.color
    )
    
    db.add(plotline)
    await db.flush()
    
    return PlotlineResponse.model_validate(plotline)


@router.get("/story/{story_id}", response_model=List[PlotlineListResponse])
async def list_plotlines(
    story_id: UUID,
    status_filter: Optional[PlotlineStatus] = Query(None),
    type_filter: Optional[PlotlineType] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all plotlines for a story"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    query = select(Plotline).where(Plotline.story_id == story_id)
    
    if status_filter:
        query = query.where(Plotline.status == status_filter)
    if type_filter:
        query = query.where(Plotline.type == type_filter)
    
    query = query.order_by(Plotline.importance.desc())
    
    result = await db.execute(query)
    plotlines = result.scalars().all()
    
    return [PlotlineListResponse.model_validate(p) for p in plotlines]


@router.get("/{plotline_id}", response_model=PlotlineResponse)
async def get_plotline(
    plotline_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific plotline"""
    plotline = await get_plotline_or_404(db, plotline_id)
    
    # Verify story ownership
    story = await story_service.get_story(db, plotline.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return PlotlineResponse.model_validate(plotline)


@router.patch("/{plotline_id}", response_model=PlotlineResponse)
async def update_plotline(
    plotline_id: UUID,
    updates: PlotlineUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a plotline"""
    plotline = await get_plotline_or_404(db, plotline_id)
    
    # Verify story ownership
    story = await story_service.get_story(db, plotline.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_data = updates.model_dump(exclude_unset=True)
    
    # Convert UUID lists to strings
    if "primary_characters" in update_data:
        update_data["primary_characters"] = [str(c) for c in update_data["primary_characters"]]
    if "secondary_characters" in update_data:
        update_data["secondary_characters"] = [str(c) for c in update_data["secondary_characters"]]
    if "connected_plotlines" in update_data:
        update_data["connected_plotlines"] = [str(p) for p in update_data["connected_plotlines"]]
    
    for key, value in update_data.items():
        setattr(plotline, key, value)
    
    plotline.updated_at = datetime.utcnow()
    await db.flush()
    
    return PlotlineResponse.model_validate(plotline)


@router.post("/{plotline_id}/plot-points", response_model=PlotlineResponse)
async def add_plot_point(
    plotline_id: UUID,
    plot_point: PlotPointCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a plot point to a plotline"""
    plotline = await get_plotline_or_404(db, plotline_id)
    
    # Verify story ownership
    story = await story_service.get_story(db, plotline.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    plot_points = plotline.plot_points or []
    plot_points.append({
        "chapter_id": str(plot_point.chapter_id) if plot_point.chapter_id else None,
        "description": plot_point.description,
        "order": plot_point.order
    })
    
    plotline.plot_points = sorted(plot_points, key=lambda x: x["order"])
    plotline.updated_at = datetime.utcnow()
    await db.flush()
    
    return PlotlineResponse.model_validate(plotline)


@router.post("/{plotline_id}/questions")
async def add_open_question(
    plotline_id: UUID,
    question: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add an open question to track"""
    plotline = await get_plotline_or_404(db, plotline_id)
    
    # Verify story ownership
    story = await story_service.get_story(db, plotline.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    questions = plotline.open_questions or []
    questions.append(question)
    plotline.open_questions = questions
    
    await db.flush()
    
    return {"message": "Question added", "questions": questions}


@router.delete("/{plotline_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_plotline(
    plotline_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a plotline"""
    plotline = await get_plotline_or_404(db, plotline_id)
    
    # Verify story ownership
    story = await story_service.get_story(db, plotline.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.delete(plotline)
