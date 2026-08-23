from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from prism.api.deps import get_provider
from prism.config import Settings
from prism.main import create_app
from prism.providers.base import ProviderResponse, RateLimitSnapshot


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

    async def aclose(self) -> None:
        return None

    @property
    def last_call(self) -> dict[str, Any]:
        assert self.calls, "provider was never called"
        return self.calls[-1]


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def client(provider: FakeProvider):
    settings = Settings(
        auth_disabled=True,
        anthropic_api_key="test",
        # Not connected to in these tests; trace writes fail and are swallowed by
        # design, which is itself part of what is being verified.
        database_url="postgresql+asyncpg://prism:prism@localhost:5434/prism_test",
    )
    app = create_app(settings)
    app.dependency_overrides[get_provider] = lambda: provider
    with TestClient(app) as c:
        yield c
