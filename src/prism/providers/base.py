"""Provider adapter interface.

One upstream exists today. The interface exists anyway, because failover,
hedging, and circuit breaking (weeks 10-11) are untestable code paths with a
single provider, and retrofitting an abstraction after the fact is how gateways
end up with Anthropic's response shape leaking into every layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RateLimitSnapshot:
    """Parsed `anthropic-ratelimit-*` headers.

    Providers limit requests, input tokens, and output tokens *independently*, so
    this is three limits rather than one. Reading them lets the bucket throttle
    before tripping a 429 rather than reacting to one after the fact.
    """

    requests_limit: int | None = None
    requests_remaining: int | None = None
    requests_reset: str | None = None
    input_tokens_limit: int | None = None
    input_tokens_remaining: int | None = None
    input_tokens_reset: str | None = None
    output_tokens_limit: int | None = None
    output_tokens_remaining: int | None = None
    output_tokens_reset: str | None = None
    retry_after_s: float | None = None

    def as_json(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class ProviderResponse:
    payload: dict[str, Any]
    status_code: int
    provider_request_id: str | None
    latency_ms: int
    rate_limits: RateLimitSnapshot = field(default_factory=RateLimitSnapshot)


class Provider(Protocol):
    name: str

    async def create_message(
        self, body: dict[str, Any], *, timeout_s: float | None = None
    ) -> ProviderResponse: ...

    async def aclose(self) -> None: ...
