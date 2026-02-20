"""
Token settings helpers - per-user overrides with defaults
"""
from typing import Dict, Optional, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.models.user_ai_settings import UserAiSettings
from app.models.user_api_keys import UserApiKeys
from app.runtime_settings import get_runtime_model_name

TOKEN_LIMIT_FIELDS = [
    "max_tokens_story_generation",
    "max_tokens_recap",
    "max_tokens_summary",
    "max_tokens_grammar",
    "max_tokens_branching",
    "max_tokens_story_to_image_prompt",
    "max_tokens_image_to_story",
    "max_tokens_character_extraction",
    "max_tokens_rewrite",
    "max_tokens_dialogue",
    "max_tokens_brainstorm",
    "max_tokens_story_bible",
    "max_tokens_story_bible_update",
    "max_tokens_import_story",
]


def get_default_token_limits() -> Dict[str, int]:
    return {field: int(getattr(settings, field)) for field in TOKEN_LIMIT_FIELDS}


async def get_user_ai_settings(db: AsyncSession, user_id: UUID) -> Optional[UserAiSettings]:
    result = await db.execute(select(UserAiSettings).where(UserAiSettings.user_id == user_id))
    return result.scalar_one_or_none()


async def get_user_token_limits(db: AsyncSession, user_id: UUID) -> Dict[str, int]:
    defaults = get_default_token_limits()
    overrides = await get_user_ai_settings(db, user_id)

    if not overrides:
        return defaults

    effective = defaults.copy()
    for field in TOKEN_LIMIT_FIELDS:
        value = getattr(overrides, field)
        if value is not None:
            effective[field] = int(value)

    return effective


async def get_user_ai_config(db: AsyncSession, user_id: UUID) -> Dict[str, Any]:
    """
    Returns the active AI provider config for a user.
    If the user has configured an external provider with is_active=True, returns that.
    Otherwise returns Ollama with the current runtime model.

    Shape: {
        "provider": "ollama" | "openai" | "anthropic" | "gemini",
        "api_key": str | None,
        "model": str,
    }
    """
    result = await db.execute(
        select(UserApiKeys).where(
            UserApiKeys.user_id == user_id,
            UserApiKeys.is_active == True,
        )
    )
    active_key = result.scalar_one_or_none()

    if active_key:
        from app.services.external_ai_service import DEFAULT_MODELS
        model = active_key.preferred_model or DEFAULT_MODELS.get(active_key.provider, "")
        return {
            "provider": active_key.provider,
            "api_key": active_key.api_key,
            "model": model,
        }

    return {
        "provider": "ollama",
        "api_key": None,
        "model": get_runtime_model_name(),
    }
