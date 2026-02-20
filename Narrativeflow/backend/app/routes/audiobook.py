"""
Audiobook Routes - Convert story chapters to audio narration and export as zip.
Uses the existing TTS service (Kokoro / Edge TTS) to generate per-chapter
audio files stored at a deterministic, cacheable path.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from pathlib import Path
import shutil
import io
import wave
import zipfile
import re
import logging

try:
    import lameenc
    _LAMEENC_AVAILABLE = True
except ImportError:
    _LAMEENC_AVAILABLE = False

from app.database import get_db
from app.models.user import User
from app.services.chapter_service import ChapterService
from app.services.story_service import StoryService
from app.services.tts_service import tts_service
from app.routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()
chapter_service = ChapterService()
story_service = StoryService()

BACKEND_DIR = Path(__file__).parent.parent.parent


def _audio_dir(story_id: UUID) -> Path:
    return BACKEND_DIR / "static" / "tts_audio" / "audiobook" / str(story_id)


def _audio_path(story_id: UUID, chapter_id: UUID) -> Path:
    return _audio_dir(story_id) / f"chapter_{chapter_id}.wav"


def _audio_url(story_id: UUID, chapter_id: UUID) -> str:
    return f"/static/tts_audio/audiobook/{story_id}/chapter_{chapter_id}.wav"


def _wav_to_mp3(wav_path: Path, bitrate: int = 128) -> bytes:
    """Convert a WAV file to MP3 bytes using lameenc (pure-Python, no ffmpeg needed)."""
    if not _LAMEENC_AVAILABLE:
        raise RuntimeError("lameenc is not installed. Run: pip install lameenc")
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(rate)
    enc.set_channels(channels)
    enc.set_quality(2)  # 2 = highest quality
    return enc.encode(pcm) + enc.flush()


# ── Pydantic models ───────────────────────────────────────────────────────────

class GenerateChapterRequest(BaseModel):
    voice: str = "neutral"   # male | female | neutral
    speed: float = 1.0


class AudiobookChapterInfo(BaseModel):
    id: str
    title: str
    number: int
    word_count: int
    has_audio: bool
    audio_url: Optional[str]
    estimated_minutes: float


class AudiobookManifest(BaseModel):
    story_id: str
    story_title: str
    chapters: List[AudiobookChapterInfo]
    total_chapters: int
    chapters_with_audio: int


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/{story_id}", response_model=AudiobookManifest)
async def get_audiobook_manifest(
    story_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the audiobook manifest — chapter list with audio status."""
    story = await story_service.get_story(db, story_id)
    if not story or str(story.author_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Story not found")

    chapters = await chapter_service.get_chapters_by_story(db, story_id)

    chapter_list: List[AudiobookChapterInfo] = []
    for ch in chapters:
        ap = _audio_path(story_id, ch.id)
        wc = ch.word_count or len((ch.content or "").split())
        chapter_list.append(AudiobookChapterInfo(
            id=str(ch.id),
            title=ch.title,
            number=ch.number,
            word_count=wc,
            has_audio=ap.exists(),
            audio_url=_audio_url(story_id, ch.id) if ap.exists() else None,
            estimated_minutes=round(wc / 150, 1),  # avg 150 wpm narration
        ))

    return AudiobookManifest(
        story_id=str(story_id),
        story_title=story.title,
        chapters=chapter_list,
        total_chapters=len(chapter_list),
        chapters_with_audio=sum(1 for c in chapter_list if c.has_audio),
    )


@router.post("/{story_id}/chapter/{chapter_id}")
async def generate_chapter_audio(
    story_id: UUID,
    chapter_id: UUID,
    request: GenerateChapterRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate (or regenerate) TTS audio for a single chapter."""
    story = await story_service.get_story(db, story_id)
    if not story or str(story.author_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Story not found")

    chapter = await chapter_service.get_chapter(db, chapter_id)
    if not chapter or str(chapter.story_id) != str(story_id):
        raise HTTPException(status_code=404, detail="Chapter not found")

    if not chapter.content or not chapter.content.strip():
        raise HTTPException(status_code=400, detail="Chapter has no content to convert")

    # Prepare deterministic output path
    out_dir = _audio_dir(story_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _audio_path(story_id, chapter_id)

    # Call TTS service (saves to a uuid-named file)
    result = await tts_service.generate_speech(
        text=chapter.content,
        voice=request.voice,
        speed=request.speed,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "TTS generation failed"),
        )

    # Move the uuid-named file to our deterministic path
    generated = BACKEND_DIR / result["audio_path"].lstrip("/")
    if generated.exists():
        shutil.move(str(generated), str(out_path))
    else:
        raise HTTPException(status_code=500, detail="Generated audio file not found")

    return {
        "success": True,
        "chapter_id": str(chapter_id),
        "chapter_title": chapter.title,
        "chapter_number": chapter.number,
        "audio_url": _audio_url(story_id, chapter_id),
        "duration_seconds": result.get("duration_seconds", 0),
        "word_count": result.get("word_count", 0),
        "voice": request.voice,
        "speed": request.speed,
        "backend_used": result.get("backend_used"),
    }


@router.delete("/{story_id}/chapter/{chapter_id}")
async def delete_chapter_audio(
    story_id: UUID,
    chapter_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete cached audio for a chapter so it can be regenerated."""
    story = await story_service.get_story(db, story_id)
    if not story or str(story.author_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Story not found")

    audio = _audio_path(story_id, chapter_id)
    if audio.exists():
        audio.unlink()

    return {"success": True}


@router.get("/{story_id}/chapter/{chapter_id}/download")
async def download_chapter_audio(
    story_id: UUID,
    chapter_id: UUID,
    voice: str = "neutral",
    speed: float = 1.0,
    format: str = "wav",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate (if needed) and stream a single chapter's audio as a file download."""
    if format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    story = await story_service.get_story(db, story_id)
    if not story or str(story.author_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Story not found")

    chapter = await chapter_service.get_chapter(db, chapter_id)
    if not chapter or str(chapter.story_id) != str(story_id):
        raise HTTPException(status_code=404, detail="Chapter not found")

    if not chapter.content or not chapter.content.strip():
        raise HTTPException(status_code=400, detail="Chapter has no content to convert")

    out_dir = _audio_dir(story_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _audio_path(story_id, chapter_id)

    if not out_path.exists():
        result = await tts_service.generate_speech(
            text=chapter.content,
            voice=voice,
            speed=speed,
        )
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "TTS generation failed"),
            )
        generated = BACKEND_DIR / result["audio_path"].lstrip("/")
        if generated.exists():
            shutil.move(str(generated), str(out_path))
        else:
            raise HTTPException(status_code=500, detail="Generated audio file not found")

    safe_title = re.sub(r'[\\/*?:"<>|]', "", chapter.title).strip() or "chapter"

    if format == "mp3":
        try:
            mp3_bytes = _wav_to_mp3(out_path)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        filename = f"{chapter.number:02d} - {safe_title}.mp3"
        return StreamingResponse(
            io.BytesIO(mp3_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    filename = f"{chapter.number:02d} - {safe_title}.wav"
    return FileResponse(
        path=str(out_path),
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{story_id}/export")
async def export_audiobook_zip(
    story_id: UUID,
    voice: str = "neutral",
    speed: float = 1.0,
    format: str = "wav",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate audio for every chapter that has content, then stream a zip download.
    Already-cached chapters are reused; only missing ones are generated.
    """
    if format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    story = await story_service.get_story(db, story_id)
    if not story or str(story.author_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Story not found")

    chapters = await chapter_service.get_chapters_by_story(db, story_id)
    chapters_with_content = [c for c in chapters if c.content and c.content.strip()]

    if not chapters_with_content:
        raise HTTPException(status_code=400, detail="No chapters with content to convert")

    out_dir = _audio_dir(story_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate missing WAV files
    for chapter in chapters_with_content:
        out_path = _audio_path(story_id, chapter.id)
        if not out_path.exists():
            result = await tts_service.generate_speech(
                text=chapter.content,
                voice=voice,
                speed=speed,
            )
            if result.get("success"):
                generated = BACKEND_DIR / result["audio_path"].lstrip("/")
                if generated.exists():
                    shutil.move(str(generated), str(out_path))

    def _safe_name(title: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "", title).strip() or "chapter"

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for chapter in chapters_with_content:
            out_path = _audio_path(story_id, chapter.id)
            if not out_path.exists():
                continue
            base_name = f"{chapter.number:02d} - {_safe_name(chapter.title)}"
            if format == "mp3":
                try:
                    mp3_bytes = _wav_to_mp3(out_path)
                    zf.writestr(f"{base_name}.mp3", mp3_bytes)
                except Exception:
                    # Fall back to WAV for this chapter if conversion fails
                    zf.write(str(out_path), arcname=f"{base_name}.wav")
            else:
                zf.write(str(out_path), arcname=f"{base_name}.wav")
    zip_buffer.seek(0)

    safe_story_title = _safe_name(story.title)
    filename = f"{safe_story_title} - Audiobook ({format.upper()}).zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
