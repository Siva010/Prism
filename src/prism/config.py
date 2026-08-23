"""Typed settings. Everything comes from the environment; nothing is hardcoded."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PRISM_",
        extra="ignore",
    )

    # Upstream credentials are read without the PRISM_ prefix so the Anthropic SDK
    # and Prism agree on one variable.
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com", alias="ANTHROPIC_BASE_URL"
    )

    database_url: str = "postgresql+asyncpg://prism:prism@localhost:5434/prism"
    redis_url: str = "redis://localhost:6380/0"

    default_model: str = "claude-opus-5"
    default_max_tokens: int = 4096

    # Split budgets: a stream that emits its first token in 400ms but runs 45s in
    # total is healthy; 30s to first token is not. One timeout cannot say that.
    request_timeout_s: float = 600.0
    first_token_timeout_s: float = 30.0
    connect_timeout_s: float = 5.0

    log_level: str = "INFO"
    trace_bodies: bool = True

    # Disables the API-key check. Local development only.
    auth_disabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
