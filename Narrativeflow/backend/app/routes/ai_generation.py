"""
AI Generation Routes - Endpoints for AI story generation
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from enum import Enum
import logging
import asyncio

from app.database import get_db
from app.config import settings
from app.services.gemini_service import GeminiService
from app.services.prompt_builder import PromptBuilder
from app.services.memory_service import MemoryService
from app.services.story_service import StoryService
from app.services.chapter_service import ChapterService
from app.services.character_service import CharacterService
from app.services.image_service import image_service
from app.services.tts_service import tts_service
from app.services.ghibli_image_service import ghibli_service
from app.services.token_settings import get_user_token_limits, get_user_ai_config
from app.models.generation import WritingMode, GenerationType
from app.models.plotline import Plotline, PlotlineStatus
from app.models.story_bible import StoryBible

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
gemini_service = GeminiService()
prompt_builder = PromptBuilder()
memory_service = MemoryService()
story_service = StoryService()
chapter_service = ChapterService()
character_service = CharacterService()


# Helper functions for async data fetching
async def get_active_plotlines(db: AsyncSession, story_id: UUID) -> List:
    """Fetch active plotlines (not resolved or abandoned) for a story"""
    query = select(Plotline).where(
        Plotline.story_id == story_id,
        Plotline.status.notin_([PlotlineStatus.RESOLVED, PlotlineStatus.ABANDONED])
    )
    result = await db.execute(query)
    return result.scalars().all()


async def get_story_bible(db: AsyncSession, story_id: UUID):
    """Fetch story bible for a story with world_rules eagerly loaded"""
    query = select(StoryBible).where(StoryBible.story_id == story_id).options(
        selectinload(StoryBible.world_rules)
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()


class WritingModeEnum(str, Enum):
    AI_LEAD = "ai_lead"
    USER_LEAD = "user_lead"
    CO_AUTHOR = "co_author"


class GenerateRequest(BaseModel):
    story_id: UUID
    chapter_id: UUID
    writing_mode: WritingModeEnum = WritingModeEnum.CO_AUTHOR
    user_direction: Optional[str] = None
    word_target: int = Field(default=500, ge=50, le=3000)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RewriteRequest(BaseModel):
    story_id: UUID
    original_text: str
    instructions: str
    writing_mode: WritingModeEnum = WritingModeEnum.CO_AUTHOR


class DialogueRequest(BaseModel):
    story_id: UUID
    character_id: UUID
    scene_context: str
    dialogue_situation: str
    other_character_ids: Optional[List[UUID]] = None
    writing_mode: WritingModeEnum = WritingModeEnum.CO_AUTHOR


class BrainstormRequest(BaseModel):
    story_id: UUID
    brainstorm_type: str = Field(..., pattern="^(plot|character|scene|dialogue|conflict|ending)$")
    current_context: str
    specific_request: Optional[str] = None


class ImagePromptRequest(BaseModel):
    description: str
    image_type: str = Field(..., pattern="^(character|scene|cover)$")
    style: Optional[str] = None


class BranchingRequest(BaseModel):
    """Request for generating branching story choices"""
    story_id: UUID
    chapter_id: UUID
    num_branches: int = Field(default=3, ge=2, le=5)
    word_target: int = Field(default=150, ge=50, le=400)  # Words per branch preview
    writing_mode: WritingModeEnum = WritingModeEnum.CO_AUTHOR


class ImageToStoryRequest(BaseModel):
    """Request to generate story from an uploaded image"""
    story_id: UUID
    chapter_id: Optional[UUID] = None
    image_base64: str  # Base64 encoded image
    context: Optional[str] = None  # Additional context about the image
    word_target: int = Field(default=500, ge=100, le=2000)
    writing_mode: WritingModeEnum = WritingModeEnum.CO_AUTHOR


class StoryToImageRequest(BaseModel):
    """Request to generate an image from story content"""
    story_id: UUID
    content: str  # Text to visualize
    image_type: str = Field(default="scene", pattern="^(character|scene|cover|environment)$")
    style: Optional[str] = None  # Art style preference
    character_id: Optional[UUID] = None  # If generating character portrait
    generate_image: bool = True  # Whether to generate actual image
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    steps: int = Field(default=30, ge=10, le=50)


class TTSRequest(BaseModel):
    """Request for text-to-speech"""
    text: str
    voice: str = Field(default="neutral")  # male, female, neutral, or specific voice name
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    language: Optional[str] = None  # Language of text, auto-detect if not provided
    backend: Optional[str] = None  # piper, coqui, edge_tts, or auto-detect


@router.post("/generate")
async def generate_continuation(
    request: GenerateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate story continuation"""
    # Get story and chapter
    story = await story_service.get_story(db, request.story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    token_limits, user_ai_config = await asyncio.gather(
        get_user_token_limits(db, story.author_id),
        get_user_ai_config(db, story.author_id),
    )
    
    chapter = await chapter_service.get_chapter(db, request.chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # Get characters, plotlines, and story bible using async queries
    characters = await character_service.get_characters_by_story(db, request.story_id)
    active_plotlines = await get_active_plotlines(db, request.story_id)
    story_bible = await get_story_bible(db, request.story_id)
    
    # Get recent content
    recent_content = chapter.content[-2000:] if chapter.content else ""
    
    # Get character IDs for context retrieval
    character_ids = [str(c.id) for c in characters] if characters else []
    character_names = [c.name for c in characters] if characters else []
    
    # COMPREHENSIVE RAG RETRIEVAL
    # Retrieve from all sources: chapters, characters, story bible
    retrieved_context = []
    if recent_content:
        query_text = recent_content[-500:] if len(recent_content) > 500 else recent_content
        
        # Get all relevant context in parallel
        all_context = await memory_service.retrieve_all_relevant_context(
            story_id=str(request.story_id),
            query=query_text,
            character_ids=character_ids,
            exclude_chapter_id=str(request.chapter_id)
        )
        
        # Combine retrieved context with source labels
        for chunk in all_context.get("chapters", []):
            if chunk["score"] > 0.3:  # Only include relevant chunks
                retrieved_context.append(f"[Previous Scene] {chunk['content']}")
        
        for chunk in all_context.get("characters", []):
            if chunk["score"] > 0.3:
                char_name = chunk.get("metadata", {}).get("character_name", "Character")
                retrieved_context.append(f"[{char_name} Info] {chunk['content']}")
        
        for chunk in all_context.get("bible", []):
            if chunk["score"] > 0.3:
                bible_type = chunk.get("metadata", {}).get("type", "World")
                retrieved_context.append(f"[{bible_type.upper()}] {chunk['content']}")
        
        logger.info(f"RAG retrieved: {len(all_context.get('chapters', []))} chapters, "
                   f"{len(all_context.get('characters', []))} character entries, "
                   f"{len(all_context.get('bible', []))} bible entries")
    
    # Build prompt
    prompt_parts = prompt_builder.build_continuation_prompt(
        story=story,
        chapter=chapter,
        characters=characters,
        active_plotlines=active_plotlines,
        story_bible=story_bible,
        recent_content=recent_content,
        retrieved_context=retrieved_context,
        writing_mode=WritingMode(request.writing_mode.value),
        user_direction=request.user_direction,
        word_target=request.word_target
    )
    
    # Generate content (routes to Ollama or external provider based on user's settings)
    result = await gemini_service.generate_story_content_routed(
        user_config=user_ai_config,
        prompt=prompt_parts["user_prompt"],
        system_prompt=prompt_parts["system_prompt"],
        writing_mode=WritingMode(request.writing_mode.value),
        context=prompt_parts["context"],
        max_tokens=min(int(request.word_target * 1.3), token_limits["max_tokens_story_generation"])
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Generation failed"))
    
    # Update chapter content and save
    if chapter.content:
        chapter.content += "\n\n" + result["content"]
    else:
        chapter.content = result["content"]
    
    chapter.word_count = len(chapter.content.split())
    await db.commit()
    
    # Embed the chapter for semantic search (in background, non-blocking)
    try:
        await memory_service.embed_chapter(
            db=db,
            story_id=str(request.story_id),
            chapter_id=str(request.chapter_id),
            content=chapter.content,
            chapter_metadata={
                "title": chapter.title,
                "number": chapter.number,
                "characters": character_names  # Pass character names for metadata extraction
            }
        )
        logger.info(f"✓ Embedded chapter {request.chapter_id} in vector database")
    except Exception as e:
        logger.warning(f"Failed to embed chapter: {e}")
    
    return {
        "content": result["content"],
        "tokens_used": result.get("tokens_used"),
        "generation_time_ms": result.get("generation_time_ms"),
        "writing_mode": request.writing_mode.value
    }


@router.post("/generate/stream")
async def generate_continuation_stream(
    request: GenerateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate story continuation with streaming response (Server-Sent Events)"""
    # Get story and chapter
    story = await story_service.get_story(db, request.story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    token_limits = await get_user_token_limits(db, story.author_id)
    
    chapter = await chapter_service.get_chapter(db, request.chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # Get characters, plotlines, and story bible using async queries
    characters = await character_service.get_characters_by_story(db, request.story_id)
    active_plotlines = await get_active_plotlines(db, request.story_id)
    story_bible = await get_story_bible(db, request.story_id)
    
    recent_content = chapter.content[-2000:] if chapter.content else ""
    character_ids = [str(c.id) for c in characters] if characters else []
    
    # RAG retrieval for context
    retrieved_context = []
    if recent_content:
        query_text = recent_content[-500:] if len(recent_content) > 500 else recent_content
        try:
            all_context = await memory_service.retrieve_all_relevant_context(
                story_id=str(request.story_id),
                query=query_text,
                character_ids=character_ids,
                exclude_chapter_id=str(request.chapter_id)
            )
            for chunk in all_context.get("chapters", []):
                if chunk["score"] > 0.3:
                    retrieved_context.append(f"[Previous Scene] {chunk['content']}")
            for chunk in all_context.get("characters", []):
                if chunk["score"] > 0.3:
                    char_name = chunk.get("metadata", {}).get("character_name", "Character")
                    retrieved_context.append(f"[{char_name} Info] {chunk['content']}")
            for chunk in all_context.get("bible", []):
                if chunk["score"] > 0.3:
                    bible_type = chunk.get("metadata", {}).get("type", "World")
                    retrieved_context.append(f"[{bible_type.upper()}] {chunk['content']}")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
    
    # Build prompt
    prompt_parts = prompt_builder.build_continuation_prompt(
        story=story,
        chapter=chapter,
        characters=characters,
        active_plotlines=active_plotlines,
        story_bible=story_bible,
        recent_content=recent_content,
        retrieved_context=retrieved_context,
        writing_mode=WritingMode(request.writing_mode.value),
        user_direction=request.user_direction,
        word_target=request.word_target
    )
    
    # Store references for saving after generation
    story_id = str(request.story_id)
    chapter_id = str(request.chapter_id)
    character_names = [c.name for c in characters] if characters else []
    chapter_title = chapter.title
    chapter_number = chapter.number
    existing_content = chapter.content or ""
    
    async def generate_sse():
        """Generate Server-Sent Events for streaming"""
        generated_text = []
        
        try:
            async for chunk in gemini_service.generate_story_content_stream(
                prompt=prompt_parts["user_prompt"],
                system_prompt=prompt_parts["system_prompt"],
                writing_mode=WritingMode(request.writing_mode.value),
                context=prompt_parts.get("context"),
                max_tokens=min(int(request.word_target * 1.3), token_limits["max_tokens_story_generation"])
            ):
                generated_text.append(chunk)
                # SSE format: data: <content>\n\n
                yield f"data: {chunk}\n\n"
            
            # Send completion signal
            yield f"data: [DONE]\n\n"
            
            # After streaming is complete, save the content
            full_text = "".join(generated_text)
            if full_text.strip():
                # Update chapter content in database
                from app.database import get_async_session
                async with get_async_session() as save_db:
                    chap = await chapter_service.get_chapter(save_db, UUID(chapter_id))
                    if chap:
                        if chap.content:
                            chap.content += "\n\n" + full_text
                        else:
                            chap.content = full_text
                        chap.word_count = len(chap.content.split())
                        await save_db.commit()
                        
                        # Embed the updated chapter
                        try:
                            await memory_service.embed_chapter(
                                db=save_db,
                                story_id=story_id,
                                chapter_id=chapter_id,
                                content=chap.content,
                                chapter_metadata={
                                    "title": chapter_title,
                                    "number": chapter_number,
                                    "characters": character_names
                                }
                            )
                            logger.info(f"✓ Auto-embedded chapter {chapter_id} after streaming generation")
                        except Exception as e:
                            logger.warning(f"Failed to embed chapter after streaming: {e}")
                
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/rewrite")
async def rewrite_text(
    request: RewriteRequest,
    db: AsyncSession = Depends(get_db)
):
    """Rewrite text based on instructions"""
    story = await story_service.get_story(db, request.story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")

    token_limits, user_ai_config, characters = await asyncio.gather(
        get_user_token_limits(db, story.author_id),
        get_user_ai_config(db, story.author_id),
        character_service.get_characters_by_story(db, request.story_id),
    )
    
    prompt_parts = prompt_builder.build_rewrite_prompt(
        story=story,
        original_text=request.original_text,
        instructions=request.instructions,
        characters=characters,
        writing_mode=WritingMode(request.writing_mode.value)
    )
    
    result = await gemini_service.generate_story_content_routed(
        user_config=user_ai_config,
        prompt=prompt_parts["user_prompt"],
        system_prompt=prompt_parts["system_prompt"],
        writing_mode=WritingMode(request.writing_mode.value),
        context=prompt_parts["context"],
        max_tokens=token_limits["max_tokens_rewrite"]
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Rewrite failed"))
    
    return {
        "rewritten_text": result["content"],
        "original_text": request.original_text,
        "instructions": request.instructions
    }


@router.post("/dialogue")
async def generate_dialogue(
    request: DialogueRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate character dialogue"""
    character = await character_service.get_character(db, request.character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    story = await story_service.get_story(db, character.story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    token_limits, user_ai_config = await asyncio.gather(
        get_user_token_limits(db, story.author_id),
        get_user_ai_config(db, story.author_id),
    )
    
    other_characters = []
    if request.other_character_ids:
        for char_id in request.other_character_ids:
            char = await character_service.get_character(db, char_id)
            if char:
                other_characters.append(char)
    
    prompt_parts = prompt_builder.build_dialogue_prompt(
        character=character,
        scene_context=request.scene_context,
        other_characters=other_characters,
        dialogue_situation=request.dialogue_situation,
        writing_mode=WritingMode(request.writing_mode.value)
    )
    
    result = await gemini_service.generate_story_content_routed(
        user_config=user_ai_config,
        prompt=prompt_parts["user_prompt"],
        system_prompt=prompt_parts["system_prompt"],
        writing_mode=WritingMode(request.writing_mode.value),
        max_tokens=token_limits["max_tokens_dialogue"]
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Dialogue generation failed"))
    
    return {
        "dialogue": result["content"],
        "character": character.name
    }


@router.post("/brainstorm")
async def brainstorm_ideas(
    request: BrainstormRequest,
    db: AsyncSession = Depends(get_db)
):
    """Brainstorm creative ideas"""
    story = await story_service.get_story(db, request.story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    token_limits, user_ai_config = await asyncio.gather(
        get_user_token_limits(db, story.author_id),
        get_user_ai_config(db, story.author_id),
    )
    
    prompt_parts = prompt_builder.build_brainstorm_prompt(
        story=story,
        brainstorm_type=request.brainstorm_type,
        current_context=request.current_context,
        specific_request=request.specific_request
    )
    
    result = await gemini_service.generate_story_content_routed(
        user_config=user_ai_config,
        prompt=prompt_parts["user_prompt"],
        system_prompt=prompt_parts["system_prompt"],
        writing_mode=WritingMode.AI_LEAD,
        max_tokens=token_limits["max_tokens_brainstorm"]
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Brainstorming failed"))
    
    return {
        "ideas": result["content"],
        "type": request.brainstorm_type
    }


@router.post("/image-prompt")
async def generate_image_prompt(request: ImagePromptRequest):
    """Generate structured image generation prompt"""
    result = await gemini_service.generate_image_prompt(
        description=request.description,
        image_type=request.image_type,
        style=request.style
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Image prompt generation failed"))
    
    return {
        "image_prompt": result["content"],
        "type": request.image_type
    }

# ============================================================
# BRANCHING & CHOICE-BASED STORYTELLING
# ============================================================

@router.post("/branches")
async def generate_story_branches(
    request: BranchingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate multiple possible story directions for the user to choose from.
    Returns 2-5 branching options with titles and preview text.
    Supports multi-language stories.
    """
    try:
        story = await story_service.get_story(db, request.story_id)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        
        chapter = await chapter_service.get_chapter(db, request.chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")
        
        # Get context
        characters = await character_service.get_characters_by_story(db, request.story_id)
        active_plotlines = await get_active_plotlines(db, request.story_id)
        story_bible = await get_story_bible(db, request.story_id)
        recent_content = chapter.content[-2000:] if chapter.content else ""
        
        # Language support
        story_language = story.language or "English"
        if story_language != "English":
            language_instruction = f"\n\nIMPORTANT: This story is written in {story_language}. Generate all branch previews in {story_language}. Preserve the language, style, and cultural context of the original story."
        else:
            language_instruction = ""
        
        # Build branching prompt
        character_names = [c.name for c in characters[:5]] if characters else []
        plotline_names = [p.title for p in active_plotlines[:3]] if active_plotlines else []
        
        # Safely get genre and tone
        story_genre = story.genre.value if story.genre else "general"
        story_tone = story.tone.value if story.tone else "balanced"
        
        # NEW APPROACH: Generate each branch individually for reliability
        import json
        import re
        
        preview_words = min(request.word_target, 150)  # Further reduced for speed
        
        logger.info(f"Generating {request.num_branches} branches IN PARALLEL for story in {story_language}")
        
        tones = ["tense", "romantic", "mysterious", "action", "emotional", "dark", "hopeful"]
        
        # Create async function for generating a single branch
        async def generate_single_branch(i: int) -> dict:
            branch_tone = tones[i % len(tones)]
            
            system_prompt = f"""You are a creative writer. Write story content in {story_language}. Output valid JSON."""
            
            prompt = f"""Story: "{story.title}" in {story_language}
Recent content: {recent_content[-300:] if recent_content else 'Story beginning'}

Write branch {i+1} with {branch_tone} tone.

Output this JSON with ALL fields in {story_language}:
{{
  "id": {i+1},
  "title": "Short title in {story_language}",
  "description": "Brief description in {story_language}",
  "tone": "{branch_tone}",
  "preview": "Write actual story continuation here (~{preview_words} words in {story_language})"
}}

IMPORTANT: The "preview" field must contain actual story text in {story_language}, not a description."""
            
            try:
                # Reduced tokens for speed
                branch_max_tokens = int((preview_words / 0.75) + 120)
                branch_max_tokens = min(branch_max_tokens, token_limits["max_tokens_branching"])
                
                result = await gemini_service.generate_story_content(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    writing_mode=WritingMode.CO_AUTHOR,
                    max_tokens=branch_max_tokens,
                    temperature_override=0.4
                )
                
                if not result.get("success"):
                    raise Exception("Generation failed")
                
                content = result["content"].strip()
                
                # Quick cleaning
                if "```json" in content:
                    match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                elif "```" in content:
                    match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                
                # Remove quotes
                for _ in range(2):
                    if content.startswith('"') and content.endswith('"'):
                        content = content[1:-1].replace('\\"', '"').replace('\\n', '\n')
                
                # Extract JSON
                if not content.startswith('{'):
                    json_match = re.search(r'(\{.*?"preview".*?\})', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                
                # Parse
                try:
                    branch_data = json.loads(content)
                except json.JSONDecodeError:
                    # Try fixing
                    fixed = re.sub(r',(\s*[}\]])', r'\1', content).replace("'", '"')
                    branch_data = json.loads(fixed)
                
                # Extract and validate preview
                preview_text = branch_data.get("preview", "")
                
                # Log what we got
                logger.info(f"Branch {i+1}: title='{branch_data.get('title', 'N/A')[:30]}', preview_length={len(preview_text)}")
                
                # If preview is empty or just placeholder text, try to extract from content
                if not preview_text or len(preview_text.strip()) < 20 or "story text" in preview_text.lower():
                    logger.warning(f"Branch {i+1} preview invalid or too short: '{preview_text[:50]}'")
                    
                    # Try to find any substantial text in the response
                    text_match = re.search(r'"preview"\s*:\s*"([^"]{30,})"', content, re.DOTALL)
                    if text_match:
                        preview_text = text_match.group(1).strip()
                        logger.info(f"Extracted better preview: {len(preview_text)} chars")
                    else:
                        # Use first part of recent content as context
                        preview_text = recent_content[:preview_words] if recent_content else f"Story continues..."
                        logger.warning(f"Using fallback preview")
                
                return {
                    "id": branch_data.get("id", i+1),
                    "title": branch_data.get("title", f"Branch {i+1}"),
                    "description": branch_data.get("description", "A new story direction"),
                    "tone": branch_data.get("tone", branch_tone),
                    "preview": preview_text
                }
                
            except Exception as e:
                logger.error(f"Branch {i+1} generation failed: {str(e)[:150]}")
                # Use recent content snippet as preview instead of English fallback
                fallback_preview = recent_content[:preview_words] if recent_content else "..."
                return {
                    "id": i+1,
                    "title": f"Branch {i+1}",
                    "description": "Alternative story direction",
                    "tone": branch_tone,
                    "preview": fallback_preview
                }
        
        # Generate all branches in parallel
        tasks = [generate_single_branch(i) for i in range(request.num_branches)]
        all_branches = await asyncio.gather(*tasks)
        
        logger.info(f"✓ Generated {len(all_branches)}/{request.num_branches} branches in parallel")
        
        return {
            "branches": all_branches,
            "story_id": str(request.story_id),
            "chapter_id": str(request.chapter_id)
        }
        
    except Exception as e:
        logger.error(f"Error in branch generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to generate branches: {str(e)}")


@router.post("/branches/select")
async def select_story_branch(
    story_id: UUID,
    chapter_id: UUID,
    branch_preview: str,
    db: AsyncSession = Depends(get_db)
):
    """
    User selects a branch - the preview becomes canon and is added to the chapter.
    """
    chapter = await chapter_service.get_chapter(db, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # Append the selected branch preview to the chapter
    if chapter.content:
        chapter.content += "\n\n" + branch_preview
    else:
        chapter.content = branch_preview
    
    chapter.word_count = len(chapter.content.split())
    await db.commit()
    
    # Embed the updated chapter
    try:
        await memory_service.embed_chapter(
            db=db,
            story_id=str(story_id),
            chapter_id=str(chapter_id),
            content=chapter.content,
            chapter_metadata={
                "title": chapter.title,
                "number": chapter.number
            }
        )
    except Exception as e:
        logger.warning(f"Failed to embed chapter after branch selection: {e}")
    
    return {
        "success": True,
        "chapter_id": str(chapter_id),
        "new_word_count": chapter.word_count
    }


# ============================================================
# IMAGE → STORY (Vision-based story generation)
# ============================================================

@router.post("/image-to-story")
async def generate_story_from_image(
    request: ImageToStoryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate story content based on an uploaded image.
    Uses vision model to analyze the image and create narrative content.
    """
    story = await story_service.get_story(db, request.story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Get story context
    characters = await character_service.get_characters_by_story(db, request.story_id)
    token_limits = await get_user_token_limits(db, story.author_id)

    character_names = [c.name for c in characters[:5]] if characters else []
    story_language = story.language or "English"
    
    # Build vision prompt with language support
    system_prompt = f"""You are a creative writer working on "{story.title}", a {story.genre.value} story with a {story.tone.value} tone.
Your task is to analyze an image and generate story content that incorporates elements from the image into the narrative.

The story follows these characters: {', '.join(character_names) if character_names else 'Characters not yet established'}

Writing style: {story.writing_style or 'Natural, engaging prose'}
POV: {story.pov_style}
Tense: {story.tense}

IMPORTANT: Write ALL content in {story_language}. Output ONLY plain prose text in {story_language}. NO HTML tags, NO markdown formatting."""

    context = request.context or "Incorporate this image into the story naturally."
    
    prompt = f"""Analyze the provided image and write approximately {request.word_target} words of story content IN {story_language}.

ADDITIONAL CONTEXT: {context}

Guidelines:
1. Describe what you see in the image through the story's narrative lens
2. Incorporate visual elements as settings, characters, or plot points
3. Maintain the story's established tone and style
4. Create engaging, immersive prose
5. Connect the image content to the existing story if possible

Write the story content in {story_language}:"""

    result = await gemini_service.analyze_image_for_story(
        image_base64=request.image_base64,
        prompt=prompt,
        system_prompt=system_prompt,
        writing_mode=WritingMode(request.writing_mode.value),
        language=story_language,
        max_tokens=token_limits["max_tokens_image_to_story"]
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Image analysis failed"))
    
    # If a chapter was specified, optionally append the content
    response_data = {
        "content": result["content"],
        "story_id": str(request.story_id),
        "image_description": result.get("image_description", "")
    }
    
    if request.chapter_id:
        chapter = await chapter_service.get_chapter(db, request.chapter_id)
        if chapter:
            response_data["chapter_id"] = str(request.chapter_id)
    
    return response_data


# ============================================================
# STORY → IMAGE (Generate images from story content)
# ============================================================

@router.post("/story-to-image")
async def generate_image_from_story(
    request: StoryToImageRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate an image prompt and (if configured) an actual image from story content.
    Returns a detailed image generation prompt optimized for AI image generators.
    """
    story = await story_service.get_story(db, request.story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    token_limits = await get_user_token_limits(db, story.author_id)
    
    # Get additional context based on image type
    extra_context = ""
    if request.image_type == "character" and request.character_id:
        character = await character_service.get_character(db, request.character_id)
        if character:
            extra_context = f"""
CHARACTER DETAILS:
- Name: {character.name}
- Physical Description: {character.physical_description or 'Not specified'}
- Age: {character.age or 'Not specified'}
- Role: {character.role.value if character.role else 'Not specified'}
- Distinctive Features: {', '.join(character.distinctive_features) if character.distinctive_features else 'None specified'}
"""
    
    # Build prompt for image generation prompt
    system_prompt = """You are an expert at creating detailed image generation prompts for AI art tools like Stable Diffusion, DALL-E, or Midjourney.

Your task is to convert story content into a highly detailed, visually descriptive prompt that will generate compelling imagery.

Include:
1. Subject description (who/what)
2. Setting/environment details
3. Lighting and atmosphere
4. Art style and medium
5. Camera angle/composition
6. Color palette
7. Mood and emotion

Format as a single, detailed prompt paragraph optimized for AI image generation."""

    style_instruction = f"Art style preference: {request.style}" if request.style else "Choose an appropriate art style for the genre"
    
    prompt = f"""Create a detailed image generation prompt based on this story content.

IMAGE TYPE: {request.image_type}
STORY GENRE: {story.genre.value}
STORY TONE: {story.tone.value}
{style_instruction}
{extra_context}

STORY CONTENT TO VISUALIZE:
{request.content[:2000]}

Generate a detailed image prompt:"""

    result = await gemini_service.generate_story_content(
        prompt=prompt,
        system_prompt=system_prompt,
        writing_mode=WritingMode.AI_LEAD,
        max_tokens=token_limits["max_tokens_story_to_image_prompt"]
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Image prompt generation failed"))
    
    image_prompt = result["content"]
    response_data = {
        "image_prompt": image_prompt,
        "image_type": request.image_type,
        "story_id": str(request.story_id),
        "style": request.style or "auto",
        "image_url": None,
        "image_base64": None
    }
    
    # Try to generate actual image if requested
    if request.generate_image:
        # Check if Stable Diffusion is available
        sd_status = await image_service.check_availability()
        
        if sd_status.get("available"):
            # Generate image using Stable Diffusion
            image_result = await image_service.generate_image(
                prompt=image_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                style_preset=request.style
            )
            
            if image_result.get("success"):
                response_data["image_url"] = image_result["image_path"]
                response_data["image_path"] = image_result["image_path"]  # Explicit path for gallery saving
                response_data["image_base64"] = image_result["image_base64"]
                response_data["image_metadata"] = {
                    "seed": image_result.get("seed"),
                    "steps": image_result.get("steps"),
                    "width": image_result.get("width"),
                    "height": image_result.get("height")
                }
                response_data["message"] = "Image generated successfully using local Stable Diffusion"
            else:
                response_data["image_error"] = image_result.get("error")
                response_data["message"] = f"Prompt generated but image failed: {image_result.get('error')}"
        else:
            response_data["sd_status"] = sd_status
            response_data["message"] = (
                "Image prompt generated. To enable local image generation, "
                "install and run Stable Diffusion WebUI (see setup_instructions)"
            )
            response_data["setup_instructions"] = sd_status.get("setup_instructions")
    else:
        response_data["message"] = "Image prompt generated. Copy and use with your preferred image generation service."
    
    return response_data


# ============================================================
# TEXT-TO-SPEECH
# ============================================================

@router.post("/tts/generate")
async def generate_speech(request: TTSRequest):
    """
    Generate text-to-speech audio using local TTS engines.
    
    Supports multiple backends:
    - Piper TTS (fast, high quality, runs locally)
    - Coqui TTS (Python-based, more voices)
    - Edge TTS (Microsoft online, fallback)
    
    Returns audio file that can be played directly in the browser.
    """
    # Try local TTS generation
    result = await tts_service.generate_speech(
        text=request.text,
        voice=request.voice,
        speed=request.speed,
        language=request.language,
        backend=request.backend
    )
    
    if result.get("success"):
        return {
            "success": True,
            "audio_url": result["audio_path"],
            "audio_base64": result["audio_base64"],
            "audio_format": result["audio_format"],
            "duration_seconds": result["duration_seconds"],
            "word_count": result["word_count"],
            "voice": result["voice"],
            "speed": result["speed"],
            "backend_used": result["backend_used"]
        }
    else:
        # Fallback to client-side TTS configuration
        import re
        clean_text = re.sub(r'<[^>]+>', '', request.text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        voice_config = {
            "male": {"pitch": 0.9, "voiceName": "Google UK English Male"},
            "female": {"pitch": 1.1, "voiceName": "Google UK English Female"},
            "neutral": {"pitch": 1.0, "voiceName": "Google US English"}
        }
        
        config = voice_config.get(request.voice, voice_config["neutral"])
        
        return {
            "success": False,
            "fallback_to_browser": True,
            "error": result.get("error", "Using browser TTS"),
            "text": clean_text,
            "config": {
                "rate": request.speed,
                "pitch": config["pitch"],
                "preferredVoice": config["voiceName"],
                "voice": request.voice
            },
            "word_count": len(clean_text.split()),
            "estimated_duration_seconds": len(clean_text.split()) / (150 * request.speed)
        }


@router.get("/tts/voices")
async def get_available_voices():
    """
    Get list of available TTS voice options.
    """
    voices = await tts_service.get_available_voices()
    availability = await tts_service.check_availability()
    
    return {
        "voices": voices.get("voices", []),
        "backends": availability.get("backends", []),
        "recommended_backend": availability.get("recommended"),
        "speed_range": {"min": 0.5, "max": 2.0, "default": 1.0}
    }


@router.get("/tts/status")
async def get_tts_status():
    """Check TTS service availability and configured backends."""
    availability = await tts_service.check_availability()
    return availability


@router.get("/image/status")
async def get_image_status():
    """Check image generation service availability."""
    availability = await image_service.check_availability()
    return availability


# ============================================================
# CHARACTER IMAGE GENERATION
# ============================================================

class CharacterImageRequest(BaseModel):
    """Request for generating a character portrait."""
    character_id: UUID
    scene_context: Optional[str] = None
    style: Optional[str] = "portrait"
    custom_additions: Optional[str] = None
    use_stored_seed: bool = True
    generate_image: bool = False  # Whether to actually generate or just return prompt
    width: int = 512
    height: int = 768


class SceneImageRequest(BaseModel):
    """Request for generating a scene illustration."""
    story_id: UUID
    scene_description: str
    character_ids: Optional[List[UUID]] = None
    setting: Optional[str] = None
    mood: Optional[str] = None
    time_of_day: Optional[str] = None
    style: Optional[str] = "fantasy"
    generate_image: bool = False
    width: int = 768
    height: int = 512


@router.post("/image/character-portrait")
async def generate_character_portrait(
    request: CharacterImageRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a portrait image for a character using their stored attributes.
    Maintains consistency by using the character's physical description, 
    distinguishing features, and optionally a stored seed.
    """
    from app.models.character import Character
    
    # Get character
    result = await db.execute(
        select(Character).where(Character.id == request.character_id)
    )
    character = result.scalar_one_or_none()
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Get stored seed or use random
    seed = character.image_generation_seed if request.use_stored_seed and character.image_generation_seed else -1
    
    # Build the prompt using character attributes
    prompt = image_service.build_character_prompt(
        character_name=character.name,
        physical_description=character.physical_description,
        distinguishing_features=character.distinguishing_features,
        age=character.age,
        gender=character.gender,
        occupation=character.occupation,
        scene_context=request.scene_context,
        style=request.style,
        custom_additions=request.custom_additions
    )
    
    # Use stored portrait prompt if available and no custom context
    if character.portrait_prompt and not request.scene_context and not request.custom_additions:
        prompt = character.portrait_prompt
        if request.style:
            prompt = image_service._apply_style_preset(prompt, request.style)
    
    response_data = {
        "character_id": str(character.id),
        "character_name": character.name,
        "prompt": prompt,
        "seed": seed if seed != -1 else "random",
        "style": request.style
    }
    
    # Try to generate if requested
    if request.generate_image:
        sd_status = await image_service.check_availability()
        
        if sd_status.get("available"):
            image_result = await image_service.generate_image(
                prompt=prompt,
                width=request.width,
                height=request.height,
                seed=seed,
                style_preset=request.style
            )
            
            if image_result.get("success"):
                # Save the seed for future consistency
                actual_seed = image_result.get("seed")
                if actual_seed and actual_seed != -1:
                    character.image_generation_seed = actual_seed
                    await db.commit()
                
                response_data["image_url"] = image_result["image_path"]
                response_data["image_path"] = image_result["image_path"]  # Explicit path for gallery saving
                response_data["image_base64"] = image_result["image_base64"]
                response_data["actual_seed"] = actual_seed
                response_data["message"] = "Portrait generated successfully"
            else:
                response_data["error"] = image_result.get("error")
        else:
            response_data["sd_available"] = False
            response_data["message"] = "Copy the prompt to use with your image generator"
    else:
        response_data["message"] = "Copy the prompt to use with your image generator (Midjourney, DALL-E, etc.)"
    
    return response_data


@router.post("/image/scene")
async def generate_scene_image(
    request: SceneImageRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate an illustration for a story scene with characters.
    """
    from app.models.character import Character
    
    characters = []
    if request.character_ids:
        result = await db.execute(
            select(Character).where(Character.id.in_(request.character_ids))
        )
        chars = result.scalars().all()
        characters = [
            {
                "name": c.name,
                "brief_description": f"{c.gender or ''} {c.age or ''}, {c.physical_description[:100] if c.physical_description else ''}".strip()
            }
            for c in chars
        ]
    
    # Build the scene prompt
    prompt = image_service.build_scene_prompt(
        scene_description=request.scene_description,
        characters=characters,
        setting=request.setting,
        mood=request.mood,
        time_of_day=request.time_of_day,
        style=request.style
    )
    
    response_data = {
        "story_id": str(request.story_id),
        "prompt": prompt,
        "style": request.style,
        "characters_included": [c["name"] for c in characters]
    }
    
    # Try to generate if requested
    if request.generate_image:
        sd_status = await image_service.check_availability()
        
        if sd_status.get("available"):
            image_result = await image_service.generate_image(
                prompt=prompt,
                width=request.width,
                height=request.height,
                style_preset=request.style
            )
            
            if image_result.get("success"):
                response_data["image_url"] = image_result["image_path"]
                response_data["image_path"] = image_result["image_path"]  # Explicit path for gallery saving
                response_data["image_base64"] = image_result["image_base64"]
                response_data["seed"] = image_result.get("seed")
                response_data["message"] = "Scene illustration generated successfully"
            else:
                response_data["error"] = image_result.get("error")
        else:
            response_data["sd_available"] = False
            response_data["message"] = "Copy the prompt to use with your image generator"
    else:
        response_data["message"] = "Copy the prompt to use with your image generator"
    
    return response_data


class SaveCharacterPromptRequest(BaseModel):
    character_id: UUID
    prompt: str
    seed: Optional[int] = None
    style: Optional[str] = None


@router.post("/image/save-character-prompt")
async def save_character_image_settings(
    request: SaveCharacterPromptRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Save a working image prompt and seed for a character.
    Use this after finding settings that generate a good portrait.
    """
    from app.models.character import Character
    
    result = await db.execute(
        select(Character).where(Character.id == request.character_id)
    )
    character = result.scalar_one_or_none()
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    character.portrait_prompt = request.prompt
    if request.seed is not None:
        character.image_generation_seed = request.seed
    if request.style:
        character.visual_style = request.style
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"Image settings saved for {character.name}",
        "portrait_prompt": request.prompt,
        "seed": request.seed,
        "visual_style": request.style
    }


@router.get("/image/consistency-tips")
async def get_image_consistency_tips():
    """Get tips for maintaining character consistency in generated images."""
    return image_service.get_consistency_tips()


# ============================================================
# GHIBLI-STYLE IMAGE GENERATION
# ============================================================

class GhibliImageRequest(BaseModel):
    """Request for Ghibli-style image generation."""
    prompt: str
    mood: Optional[str] = None
    time_of_day: Optional[str] = None
    width: int = 512
    height: int = 512
    seed: int = -1
    style_id: str = "ghibli"  # Default to ghibli for backward compatibility


class GhibliCharacterRequest(BaseModel):
    """Request for Ghibli-style character portrait."""
    character_id: Optional[UUID] = None
    name: Optional[str] = None
    physical_description: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    distinguishing_features: Optional[List[str]] = None
    expression: Optional[str] = "gentle"
    seed: int = -1
    style_id: str = "ghibli"  # Default to ghibli for backward compatibility


class GhibliSceneRequest(BaseModel):
    """Request for Ghibli-style scene."""
    description: str
    character_ids: Optional[List[UUID]] = None
    mood: Optional[str] = "peaceful"
    time_of_day: Optional[str] = "day"
    seed: int = -1
    style_id: str = "ghibli"  # Default to ghibli for backward compatibility


# New styled image request classes
class StyledImageRequest(BaseModel):
    """Request for styled image generation with any art style."""
    prompt: str
    style_id: str = "ghibli"
    mood: Optional[str] = None
    time_of_day: Optional[str] = None
    width: int = 512
    height: int = 512
    seed: int = -1


class StyledCharacterRequest(BaseModel):
    """Request for styled character portrait with any art style."""
    character_id: Optional[UUID] = None
    name: Optional[str] = None
    physical_description: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    distinguishing_features: Optional[List[str]] = None
    expression: Optional[str] = "neutral"
    style_id: str = "ghibli"
    background: Optional[str] = None
    seed: int = -1


class StyledSceneRequest(BaseModel):
    """Request for styled scene with any art style."""
    description: str
    character_ids: Optional[List[UUID]] = None
    mood: Optional[str] = None
    time_of_day: Optional[str] = None
    style_id: str = "ghibli"
    seed: int = -1


@router.get("/ghibli/status")
async def get_ghibli_service_status():
    """Check if Ghibli image generation service is available."""
    status = await ghibli_service.check_availability()
    return status


@router.get("/ghibli/presets")
async def get_ghibli_style_presets():
    """Get available style presets (art styles, moods, times, expressions)."""
    return ghibli_service.get_style_presets()


@router.get("/image/styles")
async def get_available_art_styles():
    """Get all available art styles for image generation."""
    return {
        "styles": ghibli_service.get_available_styles(),
        "default": "ghibli"
    }


@router.post("/ghibli/generate")
async def generate_ghibli_image(request: GhibliImageRequest):
    """
    Generate a styled image from a text prompt.
    Uses SD-Turbo for fast generation with style enhancement.
    """
    # Build enhanced styled prompt
    prompt, negative = ghibli_service.build_styled_prompt(
        subject=request.prompt,
        mood=request.mood,
        time_of_day=request.time_of_day,
        style_id=request.style_id
    )
    
    # Generate the image
    result = await ghibli_service.generate_image(
        prompt=prompt,
        negative_prompt=negative,
        width=request.width,
        height=request.height,
        seed=request.seed
    )
    
    # Add style info to result
    result["style_id"] = request.style_id
    
    return result


@router.post("/ghibli/character")
async def generate_ghibli_character(
    request: GhibliCharacterRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a styled character portrait.
    Can use character_id to load from database or provide details directly.
    """
    from app.models.character import Character
    
    name = request.name
    physical_description = request.physical_description
    age = request.age
    gender = request.gender
    distinguishing_features = request.distinguishing_features
    seed = request.seed
    
    # Load character from database if ID provided
    if request.character_id:
        result = await db.execute(
            select(Character).where(Character.id == request.character_id)
        )
        character = result.scalar_one_or_none()
        
        if character:
            name = character.name
            physical_description = physical_description or character.physical_description
            age = age or character.age
            gender = gender or character.gender
            distinguishing_features = distinguishing_features or character.distinguishing_features
            
            # Use stored seed if available and not overridden
            if seed == -1 and character.image_generation_seed:
                seed = character.image_generation_seed
    
    if not name:
        raise HTTPException(status_code=400, detail="Character name is required")
    
    # Generate the portrait using styled method
    result = await ghibli_service.generate_styled_character(
        name=name,
        physical_description=physical_description,
        personality=None,
        expression=request.expression,
        background=None,
        seed=seed,
        style_id=request.style_id
    )
    
    # Save seed to character if successful and character_id provided
    if result.get("success") and request.character_id and result.get("seed"):
        try:
            result_db = await db.execute(
                select(Character).where(Character.id == request.character_id)
            )
            character = result_db.scalar_one_or_none()
            if character:
                character.image_generation_seed = result["seed"]
                character.visual_style = request.style_id
                await db.commit()
                result["seed_saved"] = True
        except Exception as e:
            logger.warning(f"Could not save seed to character: {e}")
    
    # Add style info to result
    result["style_id"] = request.style_id
    
    return result


@router.post("/ghibli/scene")
async def generate_ghibli_scene(
    request: GhibliSceneRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a styled scene illustration.
    Can include characters from the database.
    """
    from app.models.character import Character
    
    characters = []
    
    # Load characters if IDs provided
    if request.character_ids:
        for char_id in request.character_ids[:2]:  # Limit to 2 characters
            result = await db.execute(
                select(Character).where(Character.id == char_id)
            )
            character = result.scalar_one_or_none()
            if character:
                characters.append({
                    "name": character.name,
                    "description": character.physical_description
                })
    
    # Generate the scene with style
    result = await ghibli_service.generate_scene(
        description=request.description,
        characters=characters if characters else None,
        mood=request.mood,
        time_of_day=request.time_of_day,
        seed=request.seed,
        style_id=request.style_id
    )
    
    # Add style info to result
    result["style_id"] = request.style_id
    
    return result


@router.post("/ghibli/prompt-only")
async def get_ghibli_prompt(request: GhibliImageRequest):
    """
    Get the enhanced styled prompt without generating an image.
    Useful for copying to external tools.
    """
    prompt, negative = ghibli_service.build_styled_prompt(
        subject=request.prompt,
        mood=request.mood,
        time_of_day=request.time_of_day,
        style_id=request.style_id
    )
    
    style_info = ghibli_service.ART_STYLES.get(request.style_id, ghibli_service.ART_STYLES["ghibli"])
    
    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "style": {
            "id": request.style_id,
            "name": style_info["name"]
        },
        "recommended_settings": {
            "steps": 4,
            "guidance_scale": 0.0,
            "width": request.width,
            "height": request.height
        },
        "tips": [
            "For Midjourney: Add '--niji 5' for anime styles or '--v 6' for photorealistic",
            "For DALL-E: The prompt works as-is",
            "For Stable Diffusion: Use these settings with SD-Turbo model"
        ]
    }
