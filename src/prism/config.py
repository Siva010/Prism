"""Typed settings. Everything comes from the environment; nothing is hardcoded."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

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
    anthropic_base_url: str = Field(default="https://api.anthropic.com", alias="ANTHROPIC_BASE_URL")

    database_url: str = "postgresql+asyncpg://prism:prism@localhost:5434/prism"
    redis_url: str = "redis://localhost:6380/0"

    default_model: str = "claude-opus-5"
    default_max_tokens: int = 4096

    # Split budgets: a stream that emits its first token in 400ms but runs 45s in
    # total is healthy; 30s to first token is not. One timeout cannot say that.
    request_timeout_s: float = 600.0
    first_token_timeout_s: float = 30.0
    connect_timeout_s: float = 5.0

    # A large thinking budget legitimately delays the first *visible* token, so
    # the first-token budget has to be conditioned on the routing decision rather
    # than fixed. Week 9's router will set this per request; today it applies
    # whenever the request asked for thinking.
    thinking_first_token_timeout_s: float = 120.0

    # What to do with an in-flight upstream stream when the client hangs up.
    # False (default): abandon it, which closes the connection and stops
    # generation, so the tokens are not paid for. True: keep consuming to
    # completion. Worth revisiting in week 8 — once a semantic cache exists, a
    # completed entry has value that a partial one does not, and draining buys
    # that value for the cost of the remaining tokens.
    stream_drain_on_disconnect: bool = False

    # --- Layer 1: provider prefix cache (week 7) ---
    cache_enabled: bool = True
    # "5m" or "1h". A 1h write costs 2x the base input rate instead of 1.25x, so
    # it only pays back on prefixes reused for longer than five minutes.
    cache_ttl: str = "5m"
    # Cache the conversation prefix as well as tools/system. Off by default:
    # history is per-conversation, so on single-turn traffic it is pure write
    # cost for reads that never arrive.
    cache_conversation_prefix: bool = False
    # Whether tool schemas may be treated as tenant-independent. True suits tool
    # definitions checked into the repo; a deployment that builds tool
    # descriptions from customer data must set this False, because nothing in
    # the code can detect that.
    cache_trust_tools: bool = True
    prompts_root: str = "prompts"

    # Thinking text on persisted traces: persist | redact | drop. Summarized
    # thinking is useful for debugging a bad answer, and it is also the most
    # sensitive thing in the trace, so this is a deliberate switch rather than an
    # accident of implementation.
    trace_thinking: Literal["persist", "redact", "drop"] = "persist"

    log_level: str = "INFO"
    trace_bodies: bool = True

    # Disables the API-key check. Local development only.
    auth_disabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
