"""
Memory Routes - Vector embeddings and semantic search endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID

from app.database import get_db
from app.services.memory_service import MemoryService
from app.services.chapter_service import ChapterService

router = APIRouter()

memory_service = MemoryService()
chapter_service = ChapterService()


class EmbedChapterRequest(BaseModel):
    story_id: UUID
    chapter_id: UUID


class SearchRequest(BaseModel):
    story_id: UUID
    query: str
    top_k: int = 5
    exclude_chapter_id: Optional[UUID] = None


class EmbedAllRequest(BaseModel):
    story_id: UUID


@router.post("/embed-chapter")
async def embed_chapter(
    request: EmbedChapterRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Embed a chapter's content for semantic search"""
    chapter = await chapter_service.get_chapter(db, request.chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    if not chapter.content:
        return {"message": "Chapter has no content to embed", "embedded": False}
    
    # Embed in background
    async def embed_task():
        async with db.begin():
            await memory_service.embed_chapter(
                db=db,
                story_id=str(request.story_id),
                chapter_id=str(request.chapter_id),
                content=chapter.content,
                chapter_metadata={
                    "title": chapter.title,
                    "number": chapter.number
                }
            )
    
    background_tasks.add_task(embed_task)
    
    return {
        "message": "Chapter embedding started",
        "chapter_id": str(request.chapter_id),
        "embedded": True
    }


@router.post("/embed-all")
async def embed_all_chapters(
    request: EmbedAllRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Embed all chapters in a story"""
    chapters = await chapter_service.get_story_chapters(db, request.story_id)
    
    if not chapters:
        return {"message": "No chapters found", "count": 0}
    
    chapters_with_content = [ch for ch in chapters if ch.content]
    
    async def embed_all_task():
        async with db.begin():
            for chapter in chapters_with_content:
                await memory_service.embed_chapter(
                    db=db,
                    story_id=str(request.story_id),
                    chapter_id=str(chapter.id),
                    content=chapter.content,
                    chapter_metadata={
                        "title": chapter.title,
                        "number": chapter.number
                    }
                )
    
    background_tasks.add_task(embed_all_task)
    
    return {
        "message": f"Embedding {len(chapters_with_content)} chapters",
        "count": len(chapters_with_content)
    }


@router.post("/search")
async def semantic_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Search story content semantically"""
    results = await memory_service.retrieve_relevant_context(
        story_id=str(request.story_id),
        query=request.query,
        top_k=request.top_k,
        exclude_chapter_id=str(request.exclude_chapter_id) if request.exclude_chapter_id else None
    )
    
    return {
        "results": results,
        "query": request.query,
        "count": len(results)
    }


@router.get("/context/{story_id}")
async def get_story_context(
    story_id: UUID,
    max_chunks: int = 10
):
    """Get summarized context for entire story"""
    context = await memory_service.get_story_summary_context(
        story_id=str(story_id),
        max_chunks=max_chunks
    )
    
    return {
        "context": context,
        "story_id": str(story_id)
    }
