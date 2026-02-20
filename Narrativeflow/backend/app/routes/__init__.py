# Routes Package
from app.routes.auth import router as auth_router
from app.routes.stories import router as stories_router
from app.routes.chapters import router as chapters_router
from app.routes.characters import router as characters_router
from app.routes.plotlines import router as plotlines_router
from app.routes.story_bible import router as story_bible_router
from app.routes.ai_generation import router as ai_generation_router
from app.routes.ai_tools import router as ai_tools_router
from app.routes.memory import router as memory_router
from app.routes.export import router as export_router

__all__ = [
    "auth_router",
    "stories_router",
    "chapters_router",
    "characters_router",
    "plotlines_router",
    "story_bible_router",
    "ai_generation_router",
    "ai_tools_router",
    "memory_router",
    "export_router"
]
