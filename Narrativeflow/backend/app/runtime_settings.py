"""
Runtime AI settings - in-memory global overrides
"""
from threading import Lock

from app.config import settings


_lock = Lock()
_current_model = settings.ollama_model
_current_vision_model = settings.ollama_vision_model if hasattr(settings, "ollama_vision_model") else settings.ollama_model


def get_runtime_model_name() -> str:
    with _lock:
        return _current_model


def get_runtime_vision_model_name() -> str:
    with _lock:
        return _current_vision_model


def set_runtime_model_name(model: str) -> None:
    global _current_model, _current_vision_model
    with _lock:
        _current_model = model
        _current_vision_model = model
