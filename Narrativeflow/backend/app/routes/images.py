"""
Image Gallery Routes - Manage generated images
"""
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4
from urllib.parse import urlparse
import os
import shutil

from app.database import get_db
from app.models.user import User
from app.models.story import Story
from app.models.image import GeneratedImage, ImageType
from app.routes.auth import get_current_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class ImageCreate(BaseModel):
    story_id: UUID
    character_id: Optional[UUID] = None
    image_type: str = "scene"
    title: Optional[str] = None
    description: Optional[str] = None
    file_path: str
    file_name: str
    prompt: Optional[str] = None
    style_id: Optional[str] = None
    seed: Optional[int] = None
    source_text: Optional[str] = None
    tags: List[str] = []


class ImageUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_favorite: Optional[bool] = None


class ImageResponse(BaseModel):
    id: UUID
    story_id: UUID
    character_id: Optional[UUID]
    image_type: str
    title: Optional[str]
    description: Optional[str]
    file_path: str
    file_name: str
    prompt: Optional[str]
    style_id: Optional[str]
    seed: Optional[int]
    tags: List[str]
    is_favorite: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Helper to map string to enum
def get_image_type(type_str: str) -> ImageType:
    try:
        return ImageType(type_str.lower())
    except ValueError:
        return ImageType.OTHER


@router.post("/upload", response_model=ImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    story_id: UUID = Form(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload an image file and save it to the gallery"""
    try:
        logger.info(f"Uploading image for story {story_id}, user {current_user.id}")
        
        # Verify story ownership
        result = await db.execute(select(Story).where(Story.id == story_id))
        story = result.scalar_one_or_none()
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        if story.author_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Validate content type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Ensure uploads directory exists
        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        ext = os.path.splitext(file.filename)[1] if file.filename else ".png"
        filename = f"{uuid4()}{ext}"
        file_path = os.path.join(upload_dir, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Create DB record
        image = GeneratedImage(
            story_id=story_id,
            image_type=ImageType.OTHER,
            title=file.filename,
            description="Uploaded image",
            file_path=f"/static/uploads/{filename}",
            file_name=filename,
            prompt="User upload",
            created_at=datetime.utcnow()
        )
        
        db.add(image)
        await db.commit()
        await db.refresh(image)
        
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=ImageResponse)
@router.post("", response_model=ImageResponse)
async def save_generated_image(
    image_data: ImageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Save a newly generated image to the gallery"""
    try:
        logger.info(f"Saving image for story {image_data.story_id}, user {current_user.id}")
        
        # Verify story ownership
        result = await db.execute(select(Story).where(Story.id == image_data.story_id))
        story = result.scalar_one_or_none()
        if not story:
            logger.error(f"Story {image_data.story_id} not found")
            raise HTTPException(status_code=404, detail="Story not found")
        if story.author_id != current_user.id:
            logger.error(f"User {current_user.id} does not own story {image_data.story_id}")
            raise HTTPException(status_code=403, detail="Not authorized to add images to this story")
        
        file_path = image_data.file_path.strip() if image_data.file_path else ""
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path is required")

        if file_path.startswith("http://") or file_path.startswith("https://"):
            parsed = urlparse(file_path)
            file_path = parsed.path or file_path

        file_name = image_data.file_name
        if not file_name:
            file_name = os.path.basename(file_path.rstrip("/")) or f"image_{uuid4()}.png"

        # Create image record
        image = GeneratedImage(
            story_id=image_data.story_id,
            character_id=image_data.character_id,
            image_type=get_image_type(image_data.image_type),
            title=image_data.title or f"Image {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            description=image_data.description,
            file_path=file_path,
            file_name=file_name,
            prompt=image_data.prompt,
            style_id=image_data.style_id,
            seed=image_data.seed,
            source_text=image_data.source_text,
            tags=image_data.tags,
        )
        
        db.add(image)
        await db.commit()
        await db.refresh(image)
        
        logger.info(f"✓ Saved image {image.id} for story {story.title}")
        
        return ImageResponse(
            id=image.id,
            story_id=image.story_id,
            character_id=image.character_id,
            image_type=image.image_type.value,
            title=image.title,
            description=image.description,
            file_path=image.file_path,
            file_name=image.file_name,
            prompt=image.prompt,
            style_id=image.style_id,
            seed=image.seed,
            tags=image.tags or [],
            is_favorite=bool(image.is_favorite),
            created_at=image.created_at,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save image: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")


@router.get("/story/{story_id}", response_model=List[ImageResponse])
async def get_story_images(
    story_id: UUID,
    image_type: Optional[str] = None,
    favorites_only: bool = False,
    character_id: Optional[UUID] = None,
    limit: int = Query(default=50, le=100),
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all images for a story with optional filtering"""
    # Verify story ownership
    result = await db.execute(select(Story).where(Story.id == story_id))
    story = result.scalar_one_or_none()
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Build query
    query = select(GeneratedImage).where(GeneratedImage.story_id == story_id)
    
    if image_type:
        query = query.where(GeneratedImage.image_type == get_image_type(image_type))
    
    if favorites_only:
        query = query.where(GeneratedImage.is_favorite == 1)
    
    if character_id:
        query = query.where(GeneratedImage.character_id == character_id)
    
    query = query.order_by(desc(GeneratedImage.created_at)).offset(offset).limit(limit)
    
    result = await db.execute(query)
    images = result.scalars().all()
    
    return [
        ImageResponse(
            id=img.id,
            story_id=img.story_id,
            character_id=img.character_id,
            image_type=img.image_type.value,
            title=img.title,
            description=img.description,
            file_path=img.file_path,
            file_name=img.file_name,
            prompt=img.prompt,
            style_id=img.style_id,
            seed=img.seed,
            tags=img.tags or [],
            is_favorite=bool(img.is_favorite),
            created_at=img.created_at,
        )
        for img in images
    ]


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific image"""
    result = await db.execute(select(GeneratedImage).where(GeneratedImage.id == image_id))
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Verify ownership
    result = await db.execute(select(Story).where(Story.id == image.story_id))
    story = result.scalar_one_or_none()
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return ImageResponse(
        id=image.id,
        story_id=image.story_id,
        character_id=image.character_id,
        image_type=image.image_type.value,
        title=image.title,
        description=image.description,
        file_path=image.file_path,
        file_name=image.file_name,
        prompt=image.prompt,
        style_id=image.style_id,
        seed=image.seed,
        tags=image.tags or [],
        is_favorite=bool(image.is_favorite),
        created_at=image.created_at,
    )


@router.patch("/{image_id}", response_model=ImageResponse)
async def update_image(
    image_id: UUID,
    update_data: ImageUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update image metadata (title, description, tags, favorite)"""
    result = await db.execute(select(GeneratedImage).where(GeneratedImage.id == image_id))
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Verify ownership
    result = await db.execute(select(Story).where(Story.id == image.story_id))
    story = result.scalar_one_or_none()
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Update fields
    if update_data.title is not None:
        image.title = update_data.title
    if update_data.description is not None:
        image.description = update_data.description
    if update_data.tags is not None:
        image.tags = update_data.tags
    if update_data.is_favorite is not None:
        image.is_favorite = 1 if update_data.is_favorite else 0
    
    await db.commit()
    await db.refresh(image)
    
    return ImageResponse(
        id=image.id,
        story_id=image.story_id,
        character_id=image.character_id,
        image_type=image.image_type.value,
        title=image.title,
        description=image.description,
        file_path=image.file_path,
        file_name=image.file_name,
        prompt=image.prompt,
        style_id=image.style_id,
        seed=image.seed,
        tags=image.tags or [],
        is_favorite=bool(image.is_favorite),
        created_at=image.created_at,
    )


@router.delete("/{image_id}")
async def delete_image(
    image_id: UUID,
    delete_file: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete an image from the gallery"""
    result = await db.execute(select(GeneratedImage).where(GeneratedImage.id == image_id))
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Verify ownership
    result = await db.execute(select(Story).where(Story.id == image.story_id))
    story = result.scalar_one_or_none()
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Optionally delete the file
    if delete_file:
        try:
            file_path = image.file_path.lstrip('/')
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted image file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete image file: {e}")
    
    # Delete database record
    await db.delete(image)
    await db.commit()
    
    return {"message": "Image deleted successfully"}


@router.post("/{image_id}/favorite")
async def toggle_favorite(
    image_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Toggle favorite status of an image"""
    result = await db.execute(select(GeneratedImage).where(GeneratedImage.id == image_id))
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Verify ownership
    result = await db.execute(select(Story).where(Story.id == image.story_id))
    story = result.scalar_one_or_none()
    if not story or story.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Toggle favorite
    image.is_favorite = 0 if image.is_favorite else 1
    await db.commit()
    
    return {"is_favorite": bool(image.is_favorite)}
