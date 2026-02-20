"""
Chapter Service - Business logic for chapter management
"""
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.models.chapter import Chapter, ChapterStatus
from app.models.story import Story


class ChapterService:
    """Service for managing chapters"""
    
    async def create_chapter(
        self,
        db: AsyncSession,
        story_id: UUID,
        title: str,
        content: str = "",
        outline: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Chapter:
        """Create a new chapter"""
        # Get next chapter number
        next_number = await self._get_next_chapter_number(db, story_id)
        
        chapter = Chapter(
            story_id=story_id,
            title=title,
            number=next_number,
            order=next_number,
            content=content,
            outline=outline,
            notes=notes,
            status=ChapterStatus.DRAFT
        )
        
        # Get story language for word counting
        story_query = select(Story).where(Story.id == story_id)
        story_result = await db.execute(story_query)
        story = story_result.scalar_one_or_none()
        language = story.language if story else "English"
        
        # Calculate word count with language
        chapter.calculate_word_count(language)
        
        db.add(chapter)
        await db.flush()
        
        # Update story chapter count
        await self._update_story_chapter_count(db, story_id)
        
        return chapter
    
    async def get_chapter(
        self,
        db: AsyncSession,
        chapter_id: UUID
    ) -> Optional[Chapter]:
        """Get a chapter by ID"""
        query = select(Chapter).where(Chapter.id == chapter_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_chapters_by_story(
        self,
        db: AsyncSession,
        story_id: UUID
    ) -> List[Chapter]:
        """Get all chapters for a story, ordered"""
        query = (
            select(Chapter)
            .where(Chapter.story_id == story_id)
            .order_by(Chapter.order)
        )
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def update_chapter(
        self,
        db: AsyncSession,
        chapter_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[Chapter]:
        """Update chapter fields"""
        chapter = await self.get_chapter(db, chapter_id)
        if not chapter:
            return None
        
        for key, value in updates.items():
            if hasattr(chapter, key):
                setattr(chapter, key, value)
        
        # Recalculate word count if content changed
        if "content" in updates:
            # Get story language
            story_query = select(Story).where(Story.id == chapter.story_id)
            story_result = await db.execute(story_query)
            story = story_result.scalar_one_or_none()
            language = story.language if story else "English"
            chapter.calculate_word_count(language)
        
        chapter.updated_at = datetime.utcnow()
        await db.flush()
        
        return chapter
    
    async def update_chapter_content(
        self,
        db: AsyncSession,
        chapter_id: UUID,
        content: str,
        append: bool = False
    ) -> Optional[Chapter]:
        """Update chapter content specifically"""
        chapter = await self.get_chapter(db, chapter_id)
        if not chapter:
            return None
        
        if append:
            chapter.content = (chapter.content or "") + content
        else:
            chapter.content = content
        
        # Get story language
        story_query = select(Story).where(Story.id == chapter.story_id)
        story_result = await db.execute(story_query)
        story = story_result.scalar_one_or_none()
        language = story.language if story else "English"
        
        chapter.calculate_word_count(language)
        chapter.updated_at = datetime.utcnow()
        
        await db.flush()
        return chapter
    
    async def delete_chapter(
        self,
        db: AsyncSession,
        chapter_id: UUID
    ) -> bool:
        """Delete a chapter"""
        chapter = await self.get_chapter(db, chapter_id)
        if not chapter:
            return False
        
        story_id = chapter.story_id
        
        await db.delete(chapter)
        await db.flush()
        
        # Renumber remaining chapters
        await self._renumber_chapters(db, story_id)
        await self._update_story_chapter_count(db, story_id)
        
        return True
    
    async def reorder_chapters(
        self,
        db: AsyncSession,
        story_id: UUID,
        chapter_order: List[UUID]
    ) -> List[Chapter]:
        """Reorder chapters in a story"""
        chapters = await self.get_chapters_by_story(db, story_id)
        chapter_map = {c.id: c for c in chapters}
        
        for index, chapter_id in enumerate(chapter_order):
            if chapter_id in chapter_map:
                chapter_map[chapter_id].order = index + 1
                chapter_map[chapter_id].number = index + 1
        
        await db.flush()
        return await self.get_chapters_by_story(db, story_id)
    
    async def get_recent_content(
        self,
        db: AsyncSession,
        chapter_id: UUID,
        word_limit: int = 1000
    ) -> str:
        """Get recent content from chapter for context"""
        chapter = await self.get_chapter(db, chapter_id)
        if not chapter or not chapter.content:
            return ""
        
        # Get last N words
        words = chapter.content.split()
        if len(words) <= word_limit:
            return chapter.content
        
        return " ".join(words[-word_limit:])
    
    async def get_chapter_context(
        self,
        db: AsyncSession,
        chapter_id: UUID,
        include_previous: int = 1
    ) -> Dict[str, Any]:
        """Get chapter with surrounding context"""
        chapter = await self.get_chapter(db, chapter_id)
        if not chapter:
            return {}
        
        # Get previous chapters
        previous = []
        if include_previous > 0:
            query = (
                select(Chapter)
                .where(Chapter.story_id == chapter.story_id)
                .where(Chapter.order < chapter.order)
                .order_by(Chapter.order.desc())
                .limit(include_previous)
            )
            result = await db.execute(query)
            previous = list(result.scalars().all())
        
        return {
            "chapter": chapter,
            "previous_chapters": list(reversed(previous))
        }
    
    async def _get_next_chapter_number(self, db: AsyncSession, story_id: UUID) -> int:
        """Get the next chapter number for a story"""
        query = select(func.max(Chapter.number)).where(Chapter.story_id == story_id)
        result = await db.execute(query)
        max_num = result.scalar() or 0
        return max_num + 1
    
    async def _renumber_chapters(self, db: AsyncSession, story_id: UUID) -> None:
        """Renumber chapters after deletion"""
        chapters = await self.get_chapters_by_story(db, story_id)
        for index, chapter in enumerate(chapters):
            chapter.number = index + 1
            chapter.order = index + 1
    
    async def _update_story_chapter_count(self, db: AsyncSession, story_id: UUID) -> None:
        """Update the chapter count on the story"""
        query = select(func.count(Chapter.id)).where(Chapter.story_id == story_id)
        result = await db.execute(query)
        count = result.scalar() or 0
        
        story_query = select(Story).where(Story.id == story_id)
        story_result = await db.execute(story_query)
        story = story_result.scalar_one_or_none()
        
        if story:
            story.chapter_count = count
