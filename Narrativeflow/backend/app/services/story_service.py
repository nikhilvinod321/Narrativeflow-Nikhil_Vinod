"""
Story Service - Business logic for story management
"""
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.models.story import Story, StoryGenre, StoryTone, StoryStatus
from app.models.chapter import Chapter
from app.models.character import Character
from app.models.plotline import Plotline
from app.models.story_bible import StoryBible


class StoryService:
    """Service for managing stories"""
    
    async def create_story(
        self,
        db: AsyncSession,
        author_id: UUID,
        title: str,
        genre: StoryGenre = StoryGenre.OTHER,
        tone: StoryTone = StoryTone.SERIOUS,
        language: str = "English",
        logline: Optional[str] = None,
        synopsis: Optional[str] = None,
        pov_style: str = "third_person_limited",
        tense: str = "past",
        setting_time: Optional[str] = None,
        setting_place: Optional[str] = None,
        writing_style: Optional[str] = None
    ) -> Story:
        """Create a new story"""
        story = Story(
            author_id=author_id,
            title=title,
            genre=genre,
            tone=tone,
            language=language,
            logline=logline,
            synopsis=synopsis,
            pov_style=pov_style,
            tense=tense,
            setting_time=setting_time,
            setting_place=setting_place,
            writing_style=writing_style,
            status=StoryStatus.DRAFT
        )
        
        db.add(story)
        await db.flush()
        
        # Create default story bible
        bible = StoryBible(story_id=story.id)
        db.add(bible)
        
        await db.flush()
        return story
    
    async def get_story(
        self,
        db: AsyncSession,
        story_id: UUID,
        include_chapters: bool = False,
        include_characters: bool = False,
        include_plotlines: bool = False
    ) -> Optional[Story]:
        """Get a story by ID with optional related data"""
        query = select(Story).where(Story.id == story_id)
        
        if include_chapters:
            query = query.options(selectinload(Story.chapters))
        if include_characters:
            query = query.options(selectinload(Story.characters))
        if include_plotlines:
            query = query.options(selectinload(Story.plotlines))
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_stories_by_author(
        self,
        db: AsyncSession,
        author_id: UUID,
        include_stats: bool = True
    ) -> List[Story]:
        """Get all stories by an author"""
        query = select(Story).where(Story.author_id == author_id).order_by(Story.updated_at.desc())
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def update_story(
        self,
        db: AsyncSession,
        story_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[Story]:
        """Update story fields"""
        story = await self.get_story(db, story_id)
        if not story:
            return None
        
        for key, value in updates.items():
            if hasattr(story, key):
                setattr(story, key, value)
        
        story.updated_at = datetime.utcnow()
        await db.flush()
        return story
    
    async def delete_story(self, db: AsyncSession, story_id: UUID) -> bool:
        """Delete a story and all related data"""
        story = await self.get_story(db, story_id)
        if not story:
            return False
        
        await db.delete(story)
        await db.flush()
        return True
    
    async def update_word_count(self, db: AsyncSession, story_id: UUID) -> int:
        """Recalculate and update story word count"""
        # Sum word counts from all chapters
        query = select(func.sum(Chapter.word_count)).where(Chapter.story_id == story_id)
        result = await db.execute(query)
        total = result.scalar() or 0
        
        await self.update_story(db, story_id, {
            "word_count": total,
            "last_written_at": datetime.utcnow()
        })
        
        return total
    
    async def get_story_stats(self, db: AsyncSession, story_id: UUID) -> Dict[str, Any]:
        """Get comprehensive story statistics"""
        story = await self.get_story(
            db, story_id,
            include_chapters=True,
            include_characters=True,
            include_plotlines=True
        )
        
        if not story:
            return {}
        
        return {
            "word_count": story.word_count,
            "chapter_count": len(story.chapters) if story.chapters else 0,
            "character_count": len(story.characters) if story.characters else 0,
            "plotline_count": len(story.plotlines) if story.plotlines else 0,
            "status": story.status.value,
            "created_at": story.created_at,
            "updated_at": story.updated_at,
            "last_written_at": story.last_written_at
        }
    
    async def get_full_story_context(
        self,
        db: AsyncSession,
        story_id: UUID
    ) -> Dict[str, Any]:
        """Get complete story context for AI generation"""
        story = await self.get_story(
            db, story_id,
            include_chapters=True,
            include_characters=True,
            include_plotlines=True
        )
        
        if not story:
            return {}
        
        # Get story bible
        bible_query = select(StoryBible).where(StoryBible.story_id == story_id)
        bible_result = await db.execute(bible_query)
        story_bible = bible_result.scalar_one_or_none()
        
        return {
            "story": story,
            "chapters": sorted(story.chapters, key=lambda c: c.order) if story.chapters else [],
            "characters": story.characters or [],
            "plotlines": story.plotlines or [],
            "story_bible": story_bible
        }
