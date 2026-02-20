"""
External AI Service - Routes generation requests to OpenAI, Anthropic, or Google Gemini.
Called when the user has configured an external provider instead of local Ollama.
"""
import httpx
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "gemini": "gemini-1.5-flash",
}

# Available models per provider shown in the settings UI
PROVIDER_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-haiku-20240307",
    ],
    "gemini": [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
    ],
}

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
}


async def generate_external(
    provider: str,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.7,
    request_timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Call an external AI provider and return a normalised result dict
    matching GeminiService.generate_story_content output format.
    """
    start_time = time.time()

    try:
        if provider == "openai":
            result = await _call_openai(api_key, model, prompt, system_prompt, max_tokens, temperature, request_timeout)
        elif provider == "anthropic":
            result = await _call_anthropic(api_key, model, prompt, system_prompt, max_tokens, temperature, request_timeout)
        elif provider == "gemini":
            result = await _call_gemini(api_key, model, prompt, system_prompt, max_tokens, temperature, request_timeout)
        else:
            return {"content": "", "error": f"Unknown provider: {provider}", "success": False,
                    "generation_time_ms": 0}

        generation_time = int((time.time() - start_time) * 1000)
        return {
            "content": result["text"],
            "tokens_used": result.get("tokens_used", 0),
            "generation_time_ms": generation_time,
            "model": model,
            "provider": provider,
            "success": True,
        }

    except Exception as e:
        err_str = f"{type(e).__name__}: {e}"
        logger.error(f"External AI error ({provider}/{model}): {err_str}")
        return {
            "content": "",
            "error": err_str,
            "success": False,
            "generation_time_ms": int((time.time() - start_time) * 1000),
        }


# ─── Provider implementations ─────────────────────────────────────────────────

async def _call_openai(api_key, model, prompt, system_prompt, max_tokens, temperature, timeout):
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return {"text": text, "tokens_used": tokens}


async def _call_anthropic(api_key, model, prompt, system_prompt, max_tokens, temperature, timeout):
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return {"text": text, "tokens_used": tokens}


async def _call_gemini(api_key, model, prompt, system_prompt, max_tokens, temperature, timeout):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        tokens = usage.get("totalTokenCount", 0)
        return {"text": text, "tokens_used": tokens}


async def validate_api_key(provider: str, api_key: str) -> Dict[str, Any]:
    """
    Quick validation call to check if an API key works.
    Returns {valid: bool, error: str|None}
    """
    try:
        result = await generate_external(
            provider=provider,
            api_key=api_key,
            model=DEFAULT_MODELS.get(provider, ""),
            prompt="Say 'ok'",
            system_prompt="You are a test assistant.",
            max_tokens=5,
            temperature=0.0,
            request_timeout=20.0,
        )
        if result["success"]:
            return {"valid": True, "error": None}
        return {"valid": False, "error": result.get("error", "Unknown error")}
    except Exception as e:
        return {"valid": False, "error": str(e)}
