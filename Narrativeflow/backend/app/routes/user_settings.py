"""
User Settings Routes - per-user AI token limits
"""
from fastapi import APIRouter, Depends, HTTPException
import httpx
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Optional, List, Any
from uuid import UUID

from app.database import get_db
from app.routes.auth import get_current_user
from app.models.user import User
from app.models.user_ai_settings import UserAiSettings
from app.models.user_api_keys import UserApiKeys
from app.services.token_settings import TOKEN_LIMIT_FIELDS, get_default_token_limits
from app.runtime_settings import get_runtime_model_name, set_runtime_model_name
from app.config import settings

router = APIRouter()


class AiTokenSettingsUpdate(BaseModel):
    max_tokens_story_generation: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_recap: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_summary: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_grammar: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_branching: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_story_to_image_prompt: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_image_to_story: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_character_extraction: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_rewrite: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_dialogue: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_brainstorm: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_story_bible: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_story_bible_update: Optional[int] = Field(default=None, ge=50, le=8000)
    max_tokens_import_story: Optional[int] = Field(default=None, ge=50, le=8000)


class AiTokenSettingsResponse(BaseModel):
    defaults: Dict[str, int]
    overrides: Dict[str, Optional[int]]
    effective: Dict[str, int]


class ModelUpdateRequest(BaseModel):
    model: str


class ModelListResponse(BaseModel):
    models: List[str]


class ModelResponse(BaseModel):
    current_model: str


def build_response(defaults: Dict[str, int], overrides: Optional[UserAiSettings]) -> AiTokenSettingsResponse:
    overrides_dict: Dict[str, Optional[int]] = {field: None for field in TOKEN_LIMIT_FIELDS}
    effective = defaults.copy()

    if overrides:
        for field in TOKEN_LIMIT_FIELDS:
            value = getattr(overrides, field)
            overrides_dict[field] = value
            if value is not None:
                effective[field] = int(value)

    return AiTokenSettingsResponse(
        defaults=defaults,
        overrides=overrides_dict,
        effective=effective
    )


@router.get("/ai", response_model=AiTokenSettingsResponse)
async def get_ai_token_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    defaults = get_default_token_limits()
    result = await db.execute(select(UserAiSettings).where(UserAiSettings.user_id == current_user.id))
    settings_row = result.scalar_one_or_none()
    return build_response(defaults, settings_row)


@router.patch("/ai", response_model=AiTokenSettingsResponse)
async def update_ai_token_settings(
    updates: AiTokenSettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    defaults = get_default_token_limits()
    result = await db.execute(select(UserAiSettings).where(UserAiSettings.user_id == current_user.id))
    settings_row = result.scalar_one_or_none()

    if not settings_row:
        settings_row = UserAiSettings(user_id=current_user.id)
        db.add(settings_row)
        await db.flush()

    update_data = updates.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(settings_row, field, value)

    await db.commit()
    await db.refresh(settings_row)

    return build_response(defaults, settings_row)


@router.get("/models", response_model=ModelListResponse)
async def list_available_models(
    current_user: User = Depends(get_current_user)
):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            return ModelListResponse(models=models)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch Ollama models: {exc}")


@router.get("/model", response_model=ModelResponse)
async def get_current_model(
    current_user: User = Depends(get_current_user)
):
    return ModelResponse(current_model=get_runtime_model_name())


@router.patch("/model", response_model=ModelResponse)
async def update_current_model(
    update: ModelUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    # Validate against Ollama model list when possible
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            available = {m.get("name") for m in data.get("models", []) if m.get("name")}
            if available and update.model not in available:
                raise HTTPException(status_code=400, detail="Model not found in Ollama")
    except HTTPException:
        raise
    except Exception:
        # If Ollama is unreachable, allow switching and let generation errors surface later.
        pass

    set_runtime_model_name(update.model)
    return ModelResponse(current_model=get_runtime_model_name())


# ─── External API provider routes ─────────────────────────────────────────────

class ApiKeyRequest(BaseModel):
    provider: str
    api_key: str
    preferred_model: Optional[str] = None
    validate: bool = True


class ProviderModelUpdate(BaseModel):
    model: str


class ActivateProviderRequest(BaseModel):
    provider: str  # "ollama" | "openai" | "anthropic" | "gemini"


class ApiProviderStatus(BaseModel):
    provider: str
    label: str
    has_key: bool
    preferred_model: Optional[str]
    is_active: bool
    available_models: List[str]


class ApiProvidersResponse(BaseModel):
    active_provider: str
    providers: List[ApiProviderStatus]
    ollama_model: str


@router.get("/api-providers", response_model=ApiProvidersResponse)
async def get_api_providers(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.services.external_ai_service import PROVIDER_MODELS, PROVIDER_LABELS, DEFAULT_MODELS

    result = await db.execute(
        select(UserApiKeys).where(UserApiKeys.user_id == current_user.id)
    )
    rows: List[UserApiKeys] = list(result.scalars().all())
    key_map = {r.provider: r for r in rows}

    active_provider = "ollama"
    for r in rows:
        if r.is_active:
            active_provider = r.provider
            break

    provider_list = []
    for provider, label in PROVIDER_LABELS.items():
        row = key_map.get(provider)
        provider_list.append(ApiProviderStatus(
            provider=provider,
            label=label,
            has_key=row is not None,
            preferred_model=row.preferred_model if row else DEFAULT_MODELS.get(provider),
            is_active=row.is_active if row else False,
            available_models=PROVIDER_MODELS.get(provider, []),
        ))

    return ApiProvidersResponse(
        active_provider=active_provider,
        providers=provider_list,
        ollama_model=get_runtime_model_name(),
    )


@router.post("/api-keys")
async def save_api_key(
    req: ApiKeyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.services.external_ai_service import validate_api_key, DEFAULT_MODELS, PROVIDER_MODELS, PROVIDER_LABELS

    if req.provider not in PROVIDER_LABELS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")

    if req.validate:
        val = await validate_api_key(req.provider, req.api_key)
        if not val["valid"]:
            raise HTTPException(status_code=400, detail=f"API key validation failed: {val['error']}")

    # Upsert
    result = await db.execute(
        select(UserApiKeys).where(
            UserApiKeys.user_id == current_user.id,
            UserApiKeys.provider == req.provider,
        )
    )
    row = result.scalar_one_or_none()
    if row:
        row.api_key = req.api_key
        if req.preferred_model:
            row.preferred_model = req.preferred_model
        elif not row.preferred_model:
            row.preferred_model = DEFAULT_MODELS.get(req.provider)
    else:
        row = UserApiKeys(
            user_id=current_user.id,
            provider=req.provider,
            api_key=req.api_key,
            preferred_model=req.preferred_model or DEFAULT_MODELS.get(req.provider),
            is_active=False,
        )
        db.add(row)

    await db.commit()
    return {"ok": True, "provider": req.provider}


@router.delete("/api-keys/{provider}")
async def delete_api_key(
    provider: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserApiKeys).where(
            UserApiKeys.user_id == current_user.id,
            UserApiKeys.provider == provider,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Key not found")

    await db.delete(row)
    await db.commit()
    return {"ok": True}


@router.patch("/api-keys/{provider}/model")
async def update_provider_model(
    provider: str,
    update: ProviderModelUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserApiKeys).where(
            UserApiKeys.user_id == current_user.id,
            UserApiKeys.provider == provider,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="No key saved for this provider")

    row.preferred_model = update.model
    await db.commit()
    return {"ok": True, "model": update.model}


@router.patch("/ai-provider/activate")
async def activate_provider(
    req: ActivateProviderRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Set a provider as active for this user. Use 'ollama' to go back to local."""
    # Deactivate all
    result = await db.execute(
        select(UserApiKeys).where(UserApiKeys.user_id == current_user.id)
    )
    rows = list(result.scalars().all())
    for r in rows:
        r.is_active = False

    if req.provider != "ollama":
        target = next((r for r in rows if r.provider == req.provider), None)
        if not target:
            raise HTTPException(status_code=404, detail=f"No key saved for {req.provider}")
        target.is_active = True

    await db.commit()
    return {"ok": True, "active_provider": req.provider}
async def reset_ai_token_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    defaults = get_default_token_limits()
    result = await db.execute(select(UserAiSettings).where(UserAiSettings.user_id == current_user.id))
    settings_row = result.scalar_one_or_none()

    if settings_row:
        await db.delete(settings_row)
        await db.commit()

    return build_response(defaults, None)


# ─── External API Key Management ──────────────────────────────────────────────

from app.services.external_ai_service import PROVIDER_MODELS, PROVIDER_LABELS, DEFAULT_MODELS, validate_api_key


class ApiKeyUpsertRequest(BaseModel):
    provider: str           # openai | anthropic | gemini
    api_key: str
    preferred_model: Optional[str] = None
    validate: bool = True   # If True, do a quick validation call before saving


class ActivateProviderRequest(BaseModel):
    provider: str           # "ollama" | "openai" | "anthropic" | "gemini"


class ApiProviderStatus(BaseModel):
    provider: str
    label: str
    has_key: bool
    preferred_model: Optional[str]
    is_active: bool
    available_models: List[str]


class ApiProvidersResponse(BaseModel):
    active_provider: str    # "ollama" or external provider
    providers: List[ApiProviderStatus]
    ollama_model: str


@router.get("/api-providers", response_model=ApiProvidersResponse)
async def get_api_providers(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current state of all external AI providers for the user."""
    result = await db.execute(
        select(UserApiKeys).where(UserApiKeys.user_id == current_user.id)
    )
    rows = {r.provider: r for r in result.scalars().all()}

    active_provider = "ollama"
    providers = []
    for p, label in PROVIDER_LABELS.items():
        row = rows.get(p)
        is_active = bool(row and row.is_active)
        if is_active:
            active_provider = p
        providers.append(ApiProviderStatus(
            provider=p,
            label=label,
            has_key=row is not None,
            preferred_model=row.preferred_model if row else DEFAULT_MODELS.get(p),
            is_active=is_active,
            available_models=PROVIDER_MODELS.get(p, []),
        ))

    return ApiProvidersResponse(
        active_provider=active_provider,
        providers=providers,
        ollama_model=get_runtime_model_name(),
    )


@router.post("/api-keys")
async def upsert_api_key(
    body: ApiKeyUpsertRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save or update an API key for a provider. Optionally validates the key first."""
    if body.provider not in PROVIDER_LABELS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {body.provider}")

    if body.validate:
        check = await validate_api_key(body.provider, body.api_key)
        if not check["valid"]:
            raise HTTPException(status_code=400, detail=f"API key validation failed: {check['error']}")

    result = await db.execute(
        select(UserApiKeys).where(
            UserApiKeys.user_id == current_user.id,
            UserApiKeys.provider == body.provider,
        )
    )
    row = result.scalar_one_or_none()

    model = body.preferred_model or DEFAULT_MODELS.get(body.provider)
    if row:
        row.api_key = body.api_key
        row.preferred_model = model
        from datetime import datetime
        row.updated_at = datetime.utcnow()
    else:
        row = UserApiKeys(
            user_id=current_user.id,
            provider=body.provider,
            api_key=body.api_key,
            preferred_model=model,
            is_active=False,
        )
        db.add(row)

    await db.commit()
    return {"ok": True, "provider": body.provider, "model": model}


@router.delete("/api-keys/{provider}")
async def delete_api_key(
    provider: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a stored API key for a provider. Deactivates the provider if it was active."""
    result = await db.execute(
        select(UserApiKeys).where(
            UserApiKeys.user_id == current_user.id,
            UserApiKeys.provider == provider,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="No key found for this provider")
    await db.delete(row)
    await db.commit()
    return {"ok": True}


@router.patch("/api-keys/{provider}/model")
async def update_provider_model(
    provider: str,
    body: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the preferred model for a saved provider key."""
    result = await db.execute(
        select(UserApiKeys).where(
            UserApiKeys.user_id == current_user.id,
            UserApiKeys.provider == provider,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="No key found for this provider")
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    row.preferred_model = model
    await db.commit()
    return {"ok": True, "model": model}


@router.patch("/ai-provider/activate")
async def activate_provider(
    body: ActivateProviderRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Set the active AI provider for this user.
    provider="ollama" deactivates all external keys.
    """
    # Deactivate all existing active keys for this user
    result = await db.execute(
        select(UserApiKeys).where(
            UserApiKeys.user_id == current_user.id,
            UserApiKeys.is_active == True,
        )
    )
    for row in result.scalars().all():
        row.is_active = False

    if body.provider != "ollama":
        if body.provider not in PROVIDER_LABELS:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {body.provider}")
        # Activate the chosen provider's row
        result2 = await db.execute(
            select(UserApiKeys).where(
                UserApiKeys.user_id == current_user.id,
                UserApiKeys.provider == body.provider,
            )
        )
        row = result2.scalar_one_or_none()
        if not row:
            raise HTTPException(
                status_code=400,
                detail=f"No API key saved for {body.provider}. Add a key first."
            )
        row.is_active = True

    await db.commit()
    return {"ok": True, "active_provider": body.provider}

