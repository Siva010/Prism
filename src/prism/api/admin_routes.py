"""Internal admin API — trace inspection and cache/cost rollups.

This is what the Next.js trace explorer (week 12) reads. Building the read side
now keeps the trace schema honest: a column nothing can query is a column nobody
verified.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from ..db.engine import session_scope
from ..db.models import Trace
from ..db.repo import get_trace, list_traces
from ..schemas.errors import ErrorKind, PrismError
from .deps import PrincipalDep

router = APIRouter()


def _summarize(trace: Trace) -> dict[str, Any]:
    return {
        "id": str(trace.id),
        "created_at": trace.created_at.isoformat(),
        "endpoint": trace.endpoint,
        "model_requested": trace.model_requested,
        "model_resolved": trace.model_resolved,
        "status": trace.status,
        "http_status": trace.http_status,
        "error_type": trace.error_type,
        "stop_reason": trace.stop_reason,
        "finish_reason": trace.finish_reason,
        "latency_ms": trace.latency_ms,
        "upstream_latency_ms": trace.upstream_latency_ms,
        "gateway_overhead_ms": (
            trace.latency_ms - trace.upstream_latency_ms
            if trace.latency_ms is not None and trace.upstream_latency_ms is not None
            else None
        ),
        "tokens": {
            "uncached_input": trace.input_tokens,
            "cached_input": trace.cache_read_input_tokens,
            "cache_write": trace.cache_creation_input_tokens,
            "output": trace.output_tokens,
            "thinking": trace.thinking_tokens,
        },
        "cost_usd": str(trace.cost_usd),
        "prompt_version": trace.prompt_version,
        "provider_request_id": trace.provider_request_id,
    }


@router.get("/traces")
async def get_traces(
    principal: PrincipalDep,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    async with session_scope() as session:
        rows = await list_traces(session, tenant_id=principal.tenant_id, limit=limit, offset=offset)
    return {"object": "list", "data": [_summarize(t) for t in rows]}


@router.get("/traces/{trace_id}")
async def get_trace_detail(trace_id: uuid.UUID, principal: PrincipalDep) -> dict[str, Any]:
    async with session_scope() as session:
        trace = await get_trace(session, trace_id)
    if trace is None or (
        principal.tenant_id is not None and trace.tenant_id != principal.tenant_id
    ):
        # Same response for "does not exist" and "belongs to another tenant" —
        # distinguishing them would leak the existence of other tenants' traffic.
        raise PrismError(ErrorKind.INVALID_REQUEST, "Trace not found.", status_code=404)

    return _summarize(trace) | {
        "cost_breakdown": trace.cost_breakdown,
        "request_body": trace.request_body,
        "upstream_request": trace.upstream_request,
        "upstream_response": trace.upstream_response,
        "response_body": trace.response_body,
        "extra": trace.extra,
    }


@router.get("/stats/cache")
async def cache_stats(principal: PrincipalDep) -> dict[str, Any]:
    """Prefix-cache behaviour, grouped by prompt version.

    The write/read ratio is the number worth looking at: persistently above ~1
    means the breakpoints sit on volatile content, so every request pays the
    write premium and nothing is ever read back — strictly worse than placing no
    breakpoint at all.
    """
    stmt = (
        select(
            Trace.model_resolved,
            Trace.prompt_version,
            func.count().label("requests"),
            func.sum(Trace.input_tokens).label("uncached_input"),
            func.sum(Trace.cache_read_input_tokens).label("cached_input"),
            func.sum(Trace.cache_creation_input_tokens).label("cache_write"),
            func.sum(Trace.output_tokens).label("output"),
            func.sum(Trace.cost_usd).label("cost_usd"),
        )
        .where(Trace.status == "ok")
        .group_by(Trace.model_resolved, Trace.prompt_version)
        .order_by(func.count().desc())
    )
    if principal.tenant_id is not None:
        stmt = stmt.where(Trace.tenant_id == principal.tenant_id)

    async with session_scope() as session:
        rows = (await session.execute(stmt)).all()

    data = []
    for row in rows:
        cached = row.cached_input or 0
        written = row.cache_write or 0
        total_input = (row.uncached_input or 0) + cached + written
        data.append(
            {
                "model": row.model_resolved,
                "prompt_version": row.prompt_version,
                "requests": row.requests,
                "uncached_input_tokens": row.uncached_input or 0,
                "cached_input_tokens": cached,
                "cache_write_tokens": written,
                "output_tokens": row.output or 0,
                "prefix_cache_hit_rate": (cached / total_input) if total_input else 0.0,
                "write_read_ratio": (written / cached) if cached else None,
                "cost_usd": str(row.cost_usd or 0),
            }
        )
    return {"object": "list", "data": data}
