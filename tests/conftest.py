from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from prism.api.deps import get_provider, get_recorder
from prism.config import Settings
from prism.main import create_app
from prism.providers.base import ProviderResponse, RateLimitSnapshot, StreamHandle
from prism.tracing.recorder import TraceDraft, TraceRecorder


def sse_lines(events: list[dict[str, Any]]) -> list[str]:
    """Render events the way the wire does: an `event:` line, a `data:` line, a blank."""
    lines: list[str] = []
    for event in events:
        lines.append(f"event: {event['type']}")
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    return lines


TEXT_STREAM: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_01ABC",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [],
            "usage": {
                "input_tokens": 12,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 1,
            },
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "ping"},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Par"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "is."},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 5},
    },
    {"type": "message_stop"},
]


class FakeProvider:
    """Records the body it was handed and replays a scripted response.

    The assertion that matters in most tests is not what came back but what Prism
    *sent* — the translated body is the week 1-2 deliverable.
    """

    name = "anthropic"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: dict[str, Any] = {
            "id": "msg_01ABC",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "Paris."}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 3,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
        self.error: Exception | None = None

    async def create_message(
        self, body: dict[str, Any], *, timeout_s: float | None = None
    ) -> ProviderResponse:
        self.calls.append(body)
        if self.error is not None:
            raise self.error
        return ProviderResponse(
            payload=self.response,
            status_code=200,
            provider_request_id="req_test",
            latency_ms=42,
            rate_limits=RateLimitSnapshot(requests_remaining=99),
        )

    # --- streaming ---------------------------------------------------------

    stream_events: list[dict[str, Any]] | None = None
    stream_open_error: Exception | None = None

    @asynccontextmanager
    async def stream_message(
        self,
        body: dict[str, Any],
        *,
        first_token_timeout_s: float | None = None,
        total_timeout_s: float | None = None,
    ) -> AsyncIterator[StreamHandle]:
        # The real provider adds this before dispatch; the fake must too, or
        # tests would not see what actually goes on the wire.
        self.calls.append({**body, "stream": True})
        if self.stream_open_error is not None:
            raise self.stream_open_error

        events = self.stream_events if self.stream_events is not None else TEXT_STREAM

        async def lines() -> AsyncIterator[str]:
            for line in sse_lines(events):
                yield line

        self.stream_closed = False
        try:
            yield StreamHandle(
                lines(),
                provider_request_id="req_test_stream",
                rate_limits=RateLimitSnapshot(requests_remaining=98),
                first_token_timeout_s=first_token_timeout_s,
                total_timeout_s=total_timeout_s,
            )
        finally:
            # Mirrors the real provider closing the upstream connection, which is
            # what actually stops generation on a client disconnect.
            self.stream_closed = True

    async def aclose(self) -> None:
        return None

    @property
    def last_call(self) -> dict[str, Any]:
        assert self.calls, "provider was never called"
        return self.calls[-1]


class CapturingRecorder(TraceRecorder):
    """Keeps drafts in memory instead of writing them.

    Trace assembly is worth asserting on directly — whether a stream was marked
    cacheable, which error type a failure recorded — and that assertion should
    not depend on a database being up.
    """

    def __init__(self) -> None:
        super().__init__(include_bodies=True)
        self.drafts: list[TraceDraft] = []

    async def record(self, draft: TraceDraft):  # type: ignore[override]
        self.drafts.append(draft)
        return None

    def record_background(self, draft: TraceDraft) -> None:
        # Synchronous, so a test can assert immediately after the response.
        self.drafts.append(draft)

    @property
    def last(self) -> TraceDraft:
        assert self.drafts, "no trace was recorded"
        return self.drafts[-1]


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def recorder() -> CapturingRecorder:
    return CapturingRecorder()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        auth_disabled=True,
        anthropic_api_key="test",
        database_url="postgresql+asyncpg://prism:prism@localhost:5434/prism_test",
    )


@pytest.fixture
def client(provider: FakeProvider, recorder: CapturingRecorder, settings: Settings):
    app = create_app(settings)
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_recorder] = lambda: recorder
    with TestClient(app) as c:
        yield c
