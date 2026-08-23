"""Trace assembly and persistence.

Traces are written after the response has been handed to the client, and a
persistence failure is logged rather than raised: an observability layer that can
fail the request it observes is worse than no observability layer.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ..cost import CostBreakdown, TokenUsage
from ..db import repo
from ..db.engine import session_scope
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class TraceDraft:
    """Everything known about one gateway request, ready to be persisted."""

    endpoint: str
    model_requested: str
    model_resolved: str
    status: str = "ok"
    ingress_format: str = "openai"
    provider: str = "anthropic"
    stream: bool = False

    tenant_id: uuid.UUID | None = None
    api_key_id: uuid.UUID | None = None

    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None

    stop_reason: str | None = None
    finish_reason: str | None = None

    latency_ms: int | None = None
    ttft_ms: int | None = None
    upstream_latency_ms: int | None = None

    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: CostBreakdown | None = None

    prompt_version: str | None = None
    provider_request_id: str | None = None

    request_body: dict[str, Any] | None = None
    upstream_request: dict[str, Any] | None = None
    upstream_response: dict[str, Any] | None = None
    response_body: dict[str, Any] | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self, *, include_bodies: bool) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": uuid.uuid4(),
            "tenant_id": self.tenant_id,
            "api_key_id": self.api_key_id,
            "endpoint": self.endpoint,
            "ingress_format": self.ingress_format,
            "provider": self.provider,
            "model_requested": self.model_requested,
            "model_resolved": self.model_resolved,
            "stream": self.stream,
            "status": self.status,
            "http_status": self.http_status,
            "error_type": self.error_type,
            "error_message": (self.error_message or None) and self.error_message[:4000],
            "stop_reason": self.stop_reason,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "upstream_latency_ms": self.upstream_latency_ms,
            "input_tokens": self.usage.input_tokens,
            "cache_read_input_tokens": self.usage.cache_read_input_tokens,
            "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
            "output_tokens": self.usage.output_tokens,
            "thinking_tokens": self.usage.thinking_tokens,
            "cost_usd": self.cost.total_usd if self.cost else Decimal(0),
            "cost_breakdown": self.cost.as_json() if self.cost else {},
            "prompt_version": self.prompt_version,
            "provider_request_id": self.provider_request_id,
            "extra": self.extra,
        }
        if include_bodies:
            row |= {
                "request_body": self.request_body,
                "upstream_request": self.upstream_request,
                "upstream_response": self.upstream_response,
                "response_body": self.response_body,
            }
        return row


class TraceRecorder:
    def __init__(self, *, include_bodies: bool = True) -> None:
        self.include_bodies = include_bodies
        self._tasks: set[asyncio.Task[uuid.UUID | None]] = set()

    async def record(self, draft: TraceDraft) -> uuid.UUID | None:
        try:
            async with session_scope() as session:
                return await repo.insert_trace(
                    session, draft.to_row(include_bodies=self.include_bodies)
                )
        except Exception as exc:  # noqa: BLE001 - never fail a request over a trace
            log.error("trace_write_failed", error=str(exc), endpoint=draft.endpoint)
            return None

    def record_background(self, draft: TraceDraft) -> None:
        """Persist without adding write latency to the client's request."""
        task = asyncio.create_task(self.record(draft))
        # Keeping a strong reference: asyncio only holds weak ones, so an
        # unreferenced task can be garbage-collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for in-flight trace writes. Called on shutdown."""
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
