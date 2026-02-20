"""
Characters Routes - CRUD operations for characters
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import Optional, List
from uuid import UUID
import re

from app.database import get_db
from app.models.user import User
from app.models.character import Character, CharacterRole, CharacterStatus
from app.services.character_service import CharacterService
from app.services.story_service import StoryService
from app.services.memory_service import MemoryService
from app.services.gemini_service import GeminiService
from app.services.chapter_service import ChapterService
from app.services.token_settings import get_user_token_limits
from app.routes.auth import get_current_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
character_service = CharacterService()
story_service = StoryService()
memory_service = MemoryService()
gemini_service = GeminiService()
chapter_service = ChapterService()


def _extract_name_candidates(text: str, max_names: int = 12) -> List[str]:
    """Fallback name extraction for English text when AI returns no characters."""
    if not text:
        return []

    stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
        "he", "she", "they", "we", "you", "i", "me", "him", "her", "them", "us",
        "his", "hers", "their", "our", "your", "my", "its",
        "in", "on", "at", "to", "for", "of", "from", "by", "with", "into", "after", "before",
        "chapter", "story", "section", "part", "narrative", "scene"
    }

    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    seen = set()
    results: List[str] = []

    for name in candidates:
        normalized = name.strip()
        lower = normalized.lower()
        if lower in stopwords or any(word in stopwords for word in lower.split()):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
        if len(results) >= max_names:
            break

    return results


# Pydantic models
class CharacterCreate(BaseModel):
    story_id: UUID
    name: str
    role: CharacterRole = CharacterRole.SUPPORTING
    full_name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    species: str = "human"
    occupation: Optional[str] = None
    physical_description: Optional[str] = None
    personality_summary: Optional[str] = None
    personality_traits: List[str] = []
    strengths: List[str] = []
    weaknesses: List[str] = []
    backstory: Optional[str] = None
    motivation: Optional[str] = None
    speaking_style: Optional[str] = None
    catchphrases: List[str] = []


class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[CharacterRole] = None
    status: Optional[CharacterStatus] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    species: Optional[str] = None
    occupation: Optional[str] = None
    physical_description: Optional[str] = None
    personality_summary: Optional[str] = None
    personality_traits: Optional[List[str]] = None
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None
    fears: Optional[List[str]] = None
    desires: Optional[List[str]] = None
    backstory: Optional[str] = None
    motivation: Optional[str] = None
    internal_conflict: Optional[str] = None
    external_conflict: Optional[str] = None
    speaking_style: Optional[str] = None
    catchphrases: Optional[List[str]] = None
    arc_description: Optional[str] = None
    portrait_url: Optional[str] = None
    portrait_prompt: Optional[str] = None
    ai_writing_notes: Optional[str] = None
    importance: Optional[int] = None


class CharacterStateUpdate(BaseModel):
    emotional_state: Optional[str] = None
    location: Optional[str] = None
    goals: Optional[List[str]] = None
    knowledge: Optional[List[str]] = None


class RelationshipCreate(BaseModel):
    related_character_id: UUID
    relationship_type: str
    description: Optional[str] = None


class CharacterResponse(BaseModel):
    id: UUID
    story_id: UUID
    name: str
    full_name: Optional[str]
    aliases: List[str]
    role: CharacterRole
    status: CharacterStatus
    importance: int
    age: Optional[str]
    gender: Optional[str]
    species: str
    occupation: Optional[str]
    physical_description: Optional[str]
    personality_summary: Optional[str]
    personality_traits: List[str]
    strengths: List[str]
    weaknesses: List[str]
    backstory: Optional[str]
    motivation: Optional[str]
    speaking_style: Optional[str]
    catchphrases: List[str]
    current_emotional_state: Optional[str]
    current_location: Optional[str]
    portrait_url: Optional[str]
    relationships: List[dict]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CharacterListResponse(BaseModel):
    id: UUID
    name: str
    role: CharacterRole
    status: CharacterStatus
    importance: int
    personality_summary: Optional[str]
    portrait_url: Optional[str]

    class Config:
        from_attributes = True


# Routes
@router.post("", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED)
async def create_character(
    character_data: CharacterCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new character"""
    # Verify story ownership
    story = await story_service.get_story(db, character_data.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    character = await character_service.create_character(
        db=db,
        **character_data.model_dump()
    )
    
    # Embed character for semantic search (RAG)
    try:
        await memory_service.embed_character(
            db=db,
            character_id=str(character.id),
            story_id=str(character.story_id),
            character_data=character_data.model_dump()
        )
        logger.info(f"✓ Embedded character {character.name} for semantic search")
    except Exception as e:
        logger.warning(f"Failed to embed character: {e}")
    
    return CharacterResponse.model_validate(character)


@router.get("/story/{story_id}", response_model=List[CharacterListResponse])
async def list_characters(
    story_id: UUID,
    role: Optional[CharacterRole] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all characters for a story"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    characters = await character_service.get_characters_by_story(db, story_id, role)
    return [CharacterListResponse.model_validate(c) for c in characters]


@router.get("/story/{story_id}/main", response_model=List[CharacterListResponse])
async def get_main_characters(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get main characters (protagonist, antagonist, deuteragonist)"""
    # Verify story ownership
    story = await story_service.get_story(db, story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    characters = await character_service.get_main_characters(db, story_id)
    return [CharacterListResponse.model_validate(c) for c in characters]


@router.get("/{character_id}", response_model=CharacterResponse)
async def get_character(
    character_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific character"""
    character = await character_service.get_character(db, character_id)
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, character.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return CharacterResponse.model_validate(character)


@router.patch("/{character_id}", response_model=CharacterResponse)
async def update_character(
    character_id: UUID,
    updates: CharacterUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a character"""
    character = await character_service.get_character(db, character_id)
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, character.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    updated = await character_service.update_character(
        db, character_id, updates.model_dump(exclude_unset=True)
    )
    
    # Re-embed character with updated data (RAG)
    try:
        char_data = {
            "name": updated.name,
            "role": updated.role.value if updated.role else None,
            "physical_description": updated.physical_description,
            "personality_summary": updated.personality_summary,
            "occupation": updated.occupation,
            "backstory": updated.backstory,
            "speaking_style": updated.speaking_style,
            "catchphrases": updated.catchphrases or [],
            "motivation": updated.motivation,
            "current_goals": updated.current_goals or []
        }
        await memory_service.embed_character(
            db=db,
            character_id=str(character_id),
            story_id=str(updated.story_id),
            character_data=char_data
        )
        logger.info(f"✓ Re-embedded character {updated.name}")
    except Exception as e:
        logger.warning(f"Failed to re-embed character: {e}")
    
    return CharacterResponse.model_validate(updated)


@router.patch("/{character_id}/state", response_model=CharacterResponse)
async def update_character_state(
    character_id: UUID,
    state: CharacterStateUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update character's current state"""
    character = await character_service.get_character(db, character_id)
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, character.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    updated = await character_service.update_character_state(
        db, character_id, **state.model_dump(exclude_unset=True)
    )
    
    return CharacterResponse.model_validate(updated)


@router.post("/{character_id}/relationships", response_model=CharacterResponse)
async def add_relationship(
    character_id: UUID,
    relationship: RelationshipCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a relationship to another character"""
    character = await character_service.get_character(db, character_id)
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, character.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Verify related character exists and is in same story
    related = await character_service.get_character(db, relationship.related_character_id)
    if not related or related.story_id != character.story_id:
        raise HTTPException(status_code=400, detail="Related character not found in this story")
    
    updated = await character_service.add_relationship(
        db,
        character_id,
        relationship.related_character_id,
        relationship.relationship_type,
        relationship.description
    )
    
    return CharacterResponse.model_validate(updated)


@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(
    character_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a character"""
    character = await character_service.get_character(db, character_id)
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Verify story ownership
    story = await story_service.get_story(db, character.story_id)
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await character_service.delete_character(db, character_id)


# Map string roles to CharacterRole enum
ROLE_MAP = {
    "protagonist": CharacterRole.PROTAGONIST,
    "antagonist": CharacterRole.ANTAGONIST,
    "supporting": CharacterRole.SUPPORTING,
    "minor": CharacterRole.MINOR,
    "mentor": CharacterRole.MENTOR,
    "love_interest": CharacterRole.LOVE_INTEREST,
    "sidekick": CharacterRole.SIDEKICK,
    "foil": CharacterRole.FOIL,
    "deuteragonist": CharacterRole.DEUTERAGONIST,
    "narrator": CharacterRole.NARRATOR,
}


@router.post("/story/{story_id}/extract-from-content")
async def extract_characters_from_story(
    story_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Automatically extract and create characters from story content.
    Uses AI to analyze chapters and identify characters.
    """
    try:
        logger.info(f"Extracting characters for story {story_id}, user {current_user.id}")
        
        # Verify story ownership
        story = await story_service.get_story(db, story_id)
        if not story:
            logger.error(f"Story {story_id} not found")
            raise HTTPException(status_code=404, detail="Story not found")
        if story.author_id != current_user.id:
            logger.error(f"User {current_user.id} does not own story {story_id}")
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Get all chapters content
        chapters = await chapter_service.get_chapters_by_story(db, story_id)
        if not chapters:
            raise HTTPException(
                status_code=400, 
                detail="Cannot extract characters without any chapter content. Write some story first!"
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
                detail="Not enough story content to extract characters. Write more first!"
            )
        
        # Get existing character names to avoid duplicates
        existing_characters = await character_service.get_characters_by_story(db, story_id)
        existing_names = [c.name.lower() for c in existing_characters]
        
        # Limit content for API
        if len(all_content) > 12000:
            all_content = all_content[:6000] + "\n\n[...middle content omitted...]\n\n" + all_content[-6000:]
        
        logger.info(f"Content ready: {len(all_content)} chars, {len(chapters)} chapters, language: {story.language}")
        
        # Extract characters using AI with language support
        token_limits = await get_user_token_limits(db, current_user.id)

        result = await gemini_service.extract_characters_from_content(
            story_content=all_content,
            story_title=story.title,
            story_genre=story.genre.value if story.genre else "general",
            existing_character_names=[c.name for c in existing_characters],
            language=story.language or "English",
            max_tokens=token_limits["max_tokens_character_extraction"]
        )
        
        logger.info(f"AI extraction result: success={result.get('success')}, parsed={result.get('parsed')}")
        
        if not result.get("success"):
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"AI extraction failed: {error_msg}")
            if 'timeout' in error_msg.lower():
                raise HTTPException(
                    status_code=504,
                    detail="AI took too long to analyze. Try with less content or try again."
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract characters: {error_msg}"
            )
        
        if not result.get("parsed") or not result.get("characters"):
            logger.warning("No characters parsed from AI result, attempting fallback")
            if (story.language or "English") == "English":
                fallback_names = _extract_name_candidates(all_content)
                if fallback_names:
                    result["characters"] = [
                        {
                            "name": name,
                            "role": "supporting",
                            "species": "human"
                        }
                        for name in fallback_names
                    ]
                    result["total_found"] = len(result["characters"])
                    result["parsed"] = True
                else:
                    return {
                        "success": True,
                        "message": "No character names found in the story content yet.",
                        "created": [],
                        "skipped": [],
                        "total_analyzed": 0
                    }
            else:
                return {
                    "success": True,
                    "message": "AI could not identify any characters. Add named characters and try again.",
                    "created": [],
                    "skipped": [],
                    "total_analyzed": 0
                }
        
        # Create characters from extracted data
        created_characters = []
        skipped = []
        
        logger.info(f"Creating {len(result['characters'])} characters")
        
        for char_data in result["characters"]:
            char_name = char_data.get("name", "").strip()
            if not char_name:
                continue
            
            # Skip if character already exists (case-insensitive)
            if char_name.lower() in existing_names:
                skipped.append(char_name)
                continue
            
            # Map role string to enum
            role_str = char_data.get("role", "supporting").lower()
            role = ROLE_MAP.get(role_str, CharacterRole.SUPPORTING)
            
            # Create the character
            try:
                new_character = await character_service.create_character(
                    db=db,
                    story_id=story_id,
                    name=char_name,
                    full_name=char_data.get("full_name"),
                    role=role,
                    age=char_data.get("age"),
                    gender=char_data.get("gender"),
                    species=char_data.get("species", "human"),
                    occupation=char_data.get("occupation"),
                    physical_description=char_data.get("physical_description"),
                    personality_summary=char_data.get("personality_summary"),
                    personality_traits=char_data.get("personality_traits", []),
                    backstory=char_data.get("backstory"),
                    motivation=char_data.get("motivation"),
                    speaking_style=char_data.get("speaking_style"),
                    distinguishing_features=char_data.get("distinguishing_features", [])
                )
                created_characters.append({
                    "id": str(new_character.id),
                    "name": new_character.name,
                    "role": new_character.role.value
                })
                existing_names.append(char_name.lower())
                logger.info(f"✓ Created character: {char_name} ({role.value})")
            except Exception as e:
                logger.warning(f"Failed to create character {char_name}: {e}")
                skipped.append(char_name)
        
        await db.commit()
        
        logger.info(f"✓ Character extraction complete: {len(created_characters)} created, {len(skipped)} skipped")
        
        return {
            "success": True,
            "message": f"Extracted {len(created_characters)} new characters from your story",
            "created": created_characters,
            "skipped": skipped,
            "total_analyzed": result.get("total_found", 0)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Character extraction failed: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to extract characters: {str(e)}")
