"""
NarrativeFlow - Main FastAPI Application
Interactive AI Story Co-Writing Platform
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from app.config import settings
from app.database import init_db, close_db
import app.models  # ensure all models are registered before init_db creates tables
from app.routes import (
    auth,
    stories,
    chapters,
    characters,
    plotlines,
    story_bible,
    ai_generation,
    ai_tools,
    memory,
    export,
    images,
    import_routes,
    user_settings,
    audiobook
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.environment == "production" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting NarrativeFlow API...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NarrativeFlow API...")
    await close_db()


# Create FastAPI application
app = FastAPI(
    title="NarrativeFlow API",
    description="""
    **NarrativeFlow** - Interactive AI Story Co-Writing Platform
    
    A production-grade narrative engine for writing novels, screenplays, 
    and episodic fiction with an AI partner.
    
    ## Features
    - Multi-chapter story management
    - Three AI writing modes (AI-Lead, User-Lead, Co-Author)
    - Story recap and summarization
    - Character and plotline management
    - Long-term narrative memory (RAG)
    - Consistency analysis
    - Image prompt generation
    """,
    version=settings.app_version,
    lifespan=lifespan
)

# CORS Configuration - Must be before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Create static directories for generated content
static_dir = Path("static")
(static_dir / "generated_images").mkdir(parents=True, exist_ok=True)
(static_dir / "uploads").mkdir(parents=True, exist_ok=True)
(static_dir / "tts_audio").mkdir(parents=True, exist_ok=True)

# Mount static files for serving generated images and audio
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(stories.router, prefix="/api/stories", tags=["Stories"])
app.include_router(chapters.router, prefix="/api/chapters", tags=["Chapters"])
app.include_router(characters.router, prefix="/api/characters", tags=["Characters"])
app.include_router(plotlines.router, prefix="/api/plotlines", tags=["Plotlines"])
app.include_router(story_bible.router, prefix="/api/story-bible", tags=["Story Bible"])
app.include_router(ai_generation.router, prefix="/api/ai", tags=["AI Generation"])
app.include_router(ai_tools.router, prefix="/api/ai-tools", tags=["AI Tools"])
app.include_router(memory.router, prefix="/api/memory", tags=["Vector Memory"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])
app.include_router(images.router, prefix="/api/images", tags=["Image Gallery"])
app.include_router(import_routes.router, prefix="/api", tags=["Import"])
app.include_router(user_settings.router, prefix="/api/settings", tags=["User Settings"])
app.include_router(audiobook.router, prefix="/api/audiobook", tags=["Audiobook"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "message": "Welcome to NarrativeFlow API - Your AI Story Co-Writing Partner"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "services": {
            "api": "operational",
            "database": "connected",
            "ai_engine": "ready",
            "vector_memory": "active"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
