"""
Application configuration using Pydantic Settings.

Reads from environment variables and .env file.
Uses @lru_cache to ensure Settings is instantiated only once.
"""

from functools import lru_cache
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment / .env file."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    APP_NAME: str = "Video RAG Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Database ─────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./chatbot.db"

    # ── AI / Agent ───────────────────────────────────────────────
    DOCUMENT_PATH: str = "result_document.json"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── CORS ─────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["*"]


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()
