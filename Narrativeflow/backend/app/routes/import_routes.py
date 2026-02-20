"""
Import Routes - Import existing manuscripts into the system
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from uuid import UUID, uuid4
from typing import Optional
import asyncio

from app.database import get_db
from app.models.user import User
from app.models.story import Story, StoryGenre, StoryTone, StoryStatus
from app.models.chapter import Chapter, ChapterStatus
from app.models.character import Character, CharacterRole, CharacterStatus
from app.models.plotline import Plotline, PlotlineType, PlotlineStatus
from app.routes.auth import get_current_user
from app.services.file_parser import FileParser
from app.services.story_extraction import StoryExtractor
from app.services.token_settings import get_user_token_limits
from app.services.memory_service import MemoryService
from app.config import settings

router = APIRouter()
memory_service = MemoryService()


@router.post("/import")
async def import_story(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    language: str = Form("English"),
    skip_ai: bool = Form(False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Import an existing manuscript file (DOCX, PDF, TXT, EPUB, RTF).
    
    1. Parses the file to extract text and chapters
    2. Uses AI to extract characters, plotlines, and themes
    3. Creates story, chapters, and related entities in database
    4. Stores content in RAG vector database
    """
    
    # Validate file format
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    file_format = FileParser.detect_format(file.filename)
    if not file_format:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported: {', '.join(FileParser.SUPPORTED_FORMATS)}"
        )
    
    # Read file content
    file_content = await file.read()
    
    if len(file_content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty"
        )
    
    # Parse file
    try:
        print(f"📖 Parsing {file.filename} ({len(file_content)} bytes)...")
        parsed_data = FileParser.parse_file(file_content, file.filename)
        print(f"✅ Successfully parsed: {len(parsed_data.get('chapters', []))} chapters, {len(parsed_data.get('content', ''))} characters")
    except ValueError as e:
        # Specific parsing errors with helpful messages
        error_msg = str(e)
        print(f"❌ Parse error: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except Exception as e:
        # Unexpected errors
        print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file. Please ensure the file is not corrupted. If the problem persists, try converting to PDF or TXT format."
        )
    
    # Use provided title or extracted title
    story_title = title or parsed_data['title']
    
    # Extract story elements using AI (optional)
    if skip_ai:
        print(f"⏭️  Skipping AI extraction (user requested fast import)")
        extracted_elements = {
            'metadata': {'genre': 'fiction', 'tone': 'serious'},
            'characters': [],
            'plotlines': [],
            'themes': []
        }
    else:
        extractor = StoryExtractor()
        try:
            token_limits = await get_user_token_limits(db, current_user.id)
            max_tokens = token_limits.get("max_tokens_import_story", settings.max_tokens_import_story)
            print(f"🤖 Starting AI extraction for '{story_title}'...")
            extracted_elements = await extractor.extract_story_elements(
                title=story_title,
                content=parsed_data['content'],
                chapters=parsed_data['chapters'],
                max_tokens=max_tokens
            )
            print(f"✅ AI extraction complete: {len(extracted_elements.get('characters', []))} characters, {len(extracted_elements.get('plotlines', []))} plotlines")
        except Exception as e:
            # Continue even if AI extraction fails
            print(f"⚠️  AI extraction failed (continuing with import): {e}")
            extracted_elements = {
                'metadata': {'genre': 'fiction', 'tone': 'serious'},
                'characters': [],
                'plotlines': [],
                'themes': []
            }
    
    # Map extracted metadata
    metadata = extracted_elements['metadata']
    
    # Map genre
    genre_map = {
        'fantasy': StoryGenre.FANTASY,
        'science fiction': StoryGenre.SCIENCE_FICTION,
        'sci-fi': StoryGenre.SCIENCE_FICTION,
        'scifi': StoryGenre.SCIENCE_FICTION,
        'mystery': StoryGenre.MYSTERY,
        'thriller': StoryGenre.THRILLER,
        'romance': StoryGenre.ROMANCE,
        'horror': StoryGenre.HORROR,
        'historical': StoryGenre.HISTORICAL,
        'adventure': StoryGenre.ADVENTURE,
        'literary': StoryGenre.LITERARY,
        'drama': StoryGenre.DRAMA,
        'comedy': StoryGenre.COMEDY,
        'crime': StoryGenre.CRIME,
        'dystopian': StoryGenre.DYSTOPIAN,
        'young adult': StoryGenre.YOUNG_ADULT,
        'western': StoryGenre.WESTERN,
    }
    genre_str = metadata.get('genre', 'fiction').lower()
    genre = StoryGenre.OTHER
    for key, value in genre_map.items():
        if key in genre_str:
            genre = value
            break
    
    # Map tone
    tone_map = {
        'dark': StoryTone.DARK,
        'light': StoryTone.LIGHT,
        'serious': StoryTone.SERIOUS,
        'humorous': StoryTone.HUMOROUS,
        'whimsical': StoryTone.WHIMSICAL,
        'gritty': StoryTone.GRITTY,
        'romantic': StoryTone.ROMANTIC,
        'suspenseful': StoryTone.SUSPENSEFUL,
        'melancholic': StoryTone.MELANCHOLIC,
        'hopeful': StoryTone.HOPEFUL,
        'epic': StoryTone.EPIC,
        'intimate': StoryTone.INTIMATE,
    }
    tone_str = metadata.get('tone', 'serious').lower()
    tone = StoryTone.SERIOUS
    for key, value in tone_map.items():
        if key in tone_str:
            tone = value
            break
    
    # Calculate word count
    word_count = FileParser.estimate_word_count(parsed_data['content'])
    
    print(f"📊 Story stats: {word_count} words, {len(parsed_data['chapters'])} chapters")
    
    # Create story in database
    story = Story(
        id=uuid4(),
        author_id=current_user.id,
        title=story_title,
        logline=metadata.get('logline', ''),
        synopsis='',
        genre=genre,
        tone=tone,
        status=StoryStatus.IN_PROGRESS,
        pov_style='third_person_limited',
        tense='past',
        language=language,
        setting_time=metadata.get('time_period', ''),
        setting_place=metadata.get('setting', ''),
        word_count=word_count,
        chapter_count=len(parsed_data['chapters']),
        tags=metadata.get('subgenres', []) + extracted_elements.get('themes', [])
    )
    
    db.add(story)
    await db.flush()  # Get story ID
    
    # Create chapters
    chapter_objects = []
    for idx, chapter_data in enumerate(parsed_data['chapters']):
        chapter = Chapter(
            id=uuid4(),
            story_id=story.id,
            title=chapter_data['title'],
            content=chapter_data['content'],
            number=idx + 1,
            order=idx,
            word_count=FileParser.estimate_word_count(chapter_data['content']),
            status=ChapterStatus.DRAFT
        )
        chapter_objects.append(chapter)
        db.add(chapter)
    
    # Create characters
    character_objects = []
    for char_data in extracted_elements.get('characters', []):
        # Map role
        role_map = {
            'protagonist': CharacterRole.PROTAGONIST,
            'antagonist': CharacterRole.ANTAGONIST,
            'supporting': CharacterRole.SUPPORTING,
        }
        role = role_map.get(char_data.get('role', '').lower(), CharacterRole.SUPPORTING)
        
        character = Character(
            id=uuid4(),
            story_id=story.id,
            name=char_data.get('name', 'Unknown'),
            role=role,
            personality_summary=char_data.get('description', ''),
            arc_description=char_data.get('arc', ''),
            status=CharacterStatus.ALIVE
        )
        character_objects.append(character)
        db.add(character)
    
    # Create plotlines
    plotline_objects = []
    for plot_data in extracted_elements.get('plotlines', []):
        # Map type
        type_map = {
            'main_plot': PlotlineType.MAIN,
            'subplot': PlotlineType.SUBPLOT,
            'romance': PlotlineType.ROMANCE,
            'mystery': PlotlineType.MYSTERY,
            'character_arc': PlotlineType.CHARACTER_ARC,
        }
        plot_type = type_map.get(plot_data.get('type', '').lower(), PlotlineType.SUBPLOT)
        
        # Map status
        status_map = {
            'unresolved': PlotlineStatus.DEVELOPING,
            'resolved': PlotlineStatus.RESOLVED,
            'ongoing': PlotlineStatus.DEVELOPING,
        }
        plot_status = status_map.get(plot_data.get('status', '').lower(), PlotlineStatus.DEVELOPING)
        
        plotline = Plotline(
            id=uuid4(),
            story_id=story.id,
            title=plot_data.get('title', 'Untitled Plot'),
            description=plot_data.get('description', ''),
            type=plot_type,
            status=plot_status
        )
        plotline_objects.append(plotline)
        db.add(plotline)
    
    # Commit all to database
    await db.commit()
    await db.refresh(story)
    
    # Store in vector database (RAG) for AI context
    try:
        # Store each chapter in ChromaDB
        for chapter in chapter_objects:
            await memory_service.add_chapter_content(
                story_id=str(story.id),
                chapter_id=str(chapter.id),
                content=chapter.content,
                title=chapter.title
            )
        
        # Store characters in vector DB
        for character in character_objects:
            char_text = f"Character: {character.name}\nRole: {character.role.value}\nDescription: {character.description}\nCharacter Arc: {character.arc}"
            await memory_service.add_character_profile(
                story_id=str(story.id),
                character_id=str(character.id),
                profile=char_text
            )
    except Exception as e:
        # Don't fail import if vector DB storage fails
        print(f"Vector DB storage error: {e}")
    
    # Return import summary
    return {
        "success": True,
        "story": {
            "id": str(story.id),
            "title": story.title,
            "genre": story.genre.value,
            "tone": story.tone.value,
            "word_count": story.word_count,
            "chapter_count": story.chapter_count
        },
        "imported": {
            "chapters": len(chapter_objects),
            "characters": len(character_objects),
            "plotlines": len(plotline_objects),
            "format": parsed_data['metadata']['format']
        },
        "extracted_metadata": metadata
    }


@router.get("/import/supported-formats")
async def get_supported_formats():
    """Get list of supported import file formats."""
    return {
        "formats": [
            {"extension": "docx", "name": "Microsoft Word", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
            {"extension": "pdf", "name": "PDF Document", "mime_type": "application/pdf"},
            {"extension": "txt", "name": "Plain Text", "mime_type": "text/plain"},
            {"extension": "epub", "name": "EPUB Ebook", "mime_type": "application/epub+zip"},
            {"extension": "rtf", "name": "Rich Text Format", "mime_type": "application/rtf"},
        ]
    }
