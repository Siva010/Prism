"""Provider adapter interface.

One upstream exists today. The interface exists anyway, because failover,
hedging, and circuit breaking (weeks 10-11) are untestable code paths with a
single provider, and retrofitting an abstraction after the fact is how gateways
end up with Anthropic's response shape leaking into every layer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..schemas.errors import ErrorKind, PrismError


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


class StreamHandle:
    """A live upstream stream, with the two timeout budgets applied.

    Takes a line iterator rather than an HTTP response so the budget logic is
    provider-agnostic and testable without a socket.

    **Why two budgets.** A response that produces its first token in 400ms and
    runs for 45 seconds is healthy; one that takes 30 seconds to produce its
    first token is not. A single timeout cannot express that difference, so
    first-token and total-duration are separate deadlines.

    The first-token deadline is *absolute*, not per-event: `ping` frames arrive
    on a healthy idle stream, and restarting the budget on each one would mean a
    stalled generation could ping forever without ever tripping it.
    """

    def __init__(
        self,
        lines: AsyncIterator[str],
        *,
        provider_request_id: str | None = None,
        rate_limits: RateLimitSnapshot | None = None,
        first_token_timeout_s: float | None = None,
        total_timeout_s: float | None = None,
    ) -> None:
        self._lines = lines
        self.provider_request_id = provider_request_id
        self.rate_limits = rate_limits or RateLimitSnapshot()
        self.first_token_timeout_s = first_token_timeout_s
        self.total_timeout_s = total_timeout_s

    async def _raw_events(self) -> AsyncIterator[dict[str, Any]]:
        """Parse SSE frames. The `event:` line is ignored — the JSON carries `type`."""
        async for line in self._lines:
            line = line.rstrip("\r")
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError as exc:
                raise PrismError(
                    ErrorKind.UPSTREAM_SERVER_ERROR,
                    f"Upstream sent a malformed SSE frame: {exc}",
                    status_code=502,
                ) from exc

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        started = loop.time()
        first_deadline = (
            started + self.first_token_timeout_s if self.first_token_timeout_s is not None else None
        )
        total_deadline = (
            started + self.total_timeout_s if self.total_timeout_s is not None else None
        )
        saw_content = False

        iterator = self._raw_events().__aiter__()
        while True:
            deadline = total_deadline
            on_first_token_budget = False
            if (
                not saw_content
                and first_deadline is not None
                and (total_deadline is None or first_deadline <= total_deadline)
            ):
                deadline, on_first_token_budget = first_deadline, True
            try:
                if deadline is None:
                    event = await iterator.__anext__()
                else:
                    async with asyncio.timeout_at(deadline):
                        event = await iterator.__anext__()
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                # Which budget fired is recorded before the await, not inferred
                # from the clock afterwards: an event loop may wake marginally
                # early, and comparing `loop.time()` to the deadline would then
                # blame the wrong budget.
                raise PrismError(
                    ErrorKind.UPSTREAM_TIMEOUT,
                    (
                        f"Upstream produced no content within the first-token "
                        f"budget of {self.first_token_timeout_s}s."
                        if on_first_token_budget
                        else f"Upstream exceeded the total response budget of "
                        f"{self.total_timeout_s}s."
                    ),
                    status_code=504,
                ) from exc

            if event.get("type") == "error":
                error = event.get("error") or {}
                # An error can arrive *mid-stream*, after headers said 200 and
                # after bytes have already reached the client. `overloaded_error`
                # here is the same event as a 529 and must classify the same way.
                kind = (
                    ErrorKind.UPSTREAM_OVERLOADED
                    if error.get("type") == "overloaded_error"
                    else ErrorKind.UPSTREAM_SERVER_ERROR
                )
                raise PrismError(
                    kind,
                    error.get("message") or "Upstream error during stream.",
                    status_code=503 if kind is ErrorKind.UPSTREAM_OVERLOADED else 502,
                )

            if event.get("type") == "content_block_delta":
                saw_content = True

            yield event


class Provider(Protocol):
    name: str

    async def create_message(
        self, body: dict[str, Any], *, timeout_s: float | None = None
    ) -> ProviderResponse: ...

    def stream_message(
        self,
        body: dict[str, Any],
        *,
        first_token_timeout_s: float | None = None,
        total_timeout_s: float | None = None,
    ) -> Any: ...

    async def aclose(self) -> None: ...
