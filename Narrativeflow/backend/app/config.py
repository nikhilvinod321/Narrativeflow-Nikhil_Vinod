"""
NarrativeFlow Configuration
Central configuration management for the application
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = "NarrativeFlow"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:Nikhil1012@localhost:5432/narrativeflow"
    database_url_sync: str = "postgresql://postgres:Nikhil1012@localhost:5432/narrativeflow"
    
    # Ollama AI
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # Keeping these for backwards compatibility
    gemini_api_key: str = ""
    gemini_model: str = "qwen2.5:7b"
    gemini_vision_model: str = "qwen2.5:7b"
    
    # JWT Authentication
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    
    # Vector Database
    chroma_persist_directory: str = "./chroma_db"
    embedding_model: str = "nomic-embed-text"  # Ollama embedding model
    embedding_dimension: int = 768  # nomic-embed-text outputs 768 dimensions
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # AI Generation Settings
    max_tokens_per_generation: int = 2000
    temperature_creative: float = 0.9
    temperature_balanced: float = 0.7
    temperature_precise: float = 0.3

    # Per-feature token limits (can be overridden via .env)
    max_tokens_story_generation: int = 900
    max_tokens_recap: int = 600
    max_tokens_summary: int = 300
    max_tokens_grammar: int = 600
    max_tokens_branching: int = 250
    max_tokens_story_to_image_prompt: int = 250
    max_tokens_image_to_story: int = 600
    max_tokens_character_extraction: int = 900
    max_tokens_rewrite: int = 500
    max_tokens_dialogue: int = 350
    max_tokens_brainstorm: int = 400
    max_tokens_story_bible: int = 400
    max_tokens_story_bible_update: int = 300
    max_tokens_import_story: int = 2000
    
    # Story Settings
    max_chapters_per_story: int = 100
    max_characters_per_story: int = 50
    max_plotlines_per_story: int = 20
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
