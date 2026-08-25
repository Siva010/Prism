"""Internal admin API — everything the dashboard reads.

Built read-side-first, from week 2 onward, on the principle that a column
nothing can query is a column nobody verified. Several of these endpoints
existed before the feature they describe: `/stats/cache` reported the
write/read ratio in week 2, five weeks before there were any breakpoints to
misplace, which meant week 7 had a number to aim at rather than a hit rate to
admire.

Every endpoint is tenant-scoped through the principal. The one exception is
provider health, which is process-wide and carries no tenant data.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import func, select, text

from ..db.engine import session_scope
from ..db.models import Trace
from ..db.repo import get_trace, list_traces
from ..schemas.errors import ErrorKind, PrismError
from .deps import ChainDep, LimiterDep, PrincipalDep, RouterDep

router = APIRouter()


def _summarize(trace: Trace) -> dict[str, Any]:
    return {
        "id": str(trace.id),
        "created_at": trace.created_at.isoformat(),
        "endpoint": trace.endpoint,
        "model_requested": trace.model_requested,
        "model_resolved": trace.model_resolved,
        "stream": trace.stream,
        "status": trace.status,
        "http_status": trace.http_status,
        "error_type": trace.error_type,
        "stop_reason": trace.stop_reason,
        "finish_reason": trace.finish_reason,
        "latency_ms": trace.latency_ms,
        "ttft_ms": trace.ttft_ms,
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


# --- traces ---------------------------------------------------------------


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


# --- caching --------------------------------------------------------------


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


@router.get("/stats/semantic-cache")
async def semantic_cache_stats(principal: PrincipalDep) -> dict[str, Any]:
    """Layer-2 outcomes, read back off the traces.

    Reported as hits against *lookups*, and alongside the near-miss distribution,
    because a hit rate without the misses it beat says nothing about whether the
    threshold is in the right place.
    """
    stmt = text("""
        SELECT
            count(*) FILTER (WHERE extra->'semantic_cache'->>'hit' = 'true')  AS hits,
            count(*) FILTER (WHERE extra ? 'semantic_cache')                  AS lookups,
            avg((extra->'semantic_cache'->>'similarity')::float)
                FILTER (WHERE extra->'semantic_cache'->>'hit' = 'true')       AS avg_hit_similarity,
            avg((extra->'semantic_cache'->>'best_similarity')::float)
                FILTER (WHERE extra->'semantic_cache'->>'hit' = 'false')
                AS avg_miss_similarity,
            sum(cost_usd) FILTER (WHERE extra->'semantic_cache'->>'hit' = 'true') AS cost_on_hits
        FROM traces
        WHERE (:tenant IS NULL OR tenant_id = CAST(:tenant AS UUID))
    """)
    async with session_scope() as session:
        row = (
            (
                await session.execute(
                    stmt,
                    {"tenant": str(principal.tenant_id) if principal.tenant_id else None},
                )
            )
            .mappings()
            .first()
        )

    stats: dict[str, Any] = dict(row) if row else {}
    lookups = int(stats.get("lookups") or 0)
    hits = int(stats.get("hits") or 0)
    return {
        "lookups": lookups,
        "hits": hits,
        "hit_rate": (hits / lookups) if lookups else 0.0,
        "avg_hit_similarity": stats.get("avg_hit_similarity"),
        # How close the misses came. This is what a threshold re-tune runs on, so
        # production traffic becomes the data rather than a separate exercise.
        "avg_miss_similarity": stats.get("avg_miss_similarity"),
        "cost_avoided_usd": str(stats.get("cost_on_hits") or 0),
    }


@router.get("/cache/calibrations")
async def cache_calibrations(
    principal: PrincipalDep, limit: int = Query(10, ge=1, le=50)
) -> dict[str, Any]:
    """Threshold calibration runs — the ROC curve behind the operating point.

    A threshold in production should always be traceable to the labelled set and
    embedder that justified it, which is why these are stored rather than printed.
    """
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    text("""
                    SELECT id, created_at, embedder, n_pairs, auc, max_false_hit_rate,
                           chosen_threshold, chosen, curve
                    FROM cache_calibrations
                    ORDER BY created_at DESC LIMIT :limit
                """),
                    {"limit": limit},
                )
            )
            .mappings()
            .all()
        )

    return {
        "object": "list",
        "data": [
            {
                "id": str(r["id"]),
                "created_at": r["created_at"].isoformat(),
                "embedder": r["embedder"],
                "n_pairs": r["n_pairs"],
                "auc": r["auc"],
                "max_false_hit_rate": r["max_false_hit_rate"],
                "chosen_threshold": r["chosen_threshold"],
                "chosen": r["chosen"],
                "curve": r["curve"],
            }
            for r in rows
        ],
    }


# --- cost -----------------------------------------------------------------


@router.get("/stats/cost")
async def cost_stats(
    principal: PrincipalDep, days: int = Query(30, ge=1, le=365)
) -> dict[str, Any]:
    """Cost attribution by day, model, prompt version, and token class.

    `cost_per_successful_task` is the headline. A tier that fails 40% of the time
    and retries on an expensive one is not cheap, and only that denominator
    shows it.
    """
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    text("""
                    SELECT day, model_resolved, prompt_version, requests, successful,
                           uncached_input_tokens, cached_input_tokens,
                           cache_write_tokens, output_tokens, thinking_tokens,
                           cost_usd, cost_per_successful_task
                    FROM cost_attribution
                    WHERE (:tenant IS NULL OR tenant_id = CAST(:tenant AS UUID))
                      AND day >= current_date - CAST(:days AS INTEGER)
                    ORDER BY day DESC, cost_usd DESC
                """),
                    {
                        "tenant": str(principal.tenant_id) if principal.tenant_id else None,
                        "days": days,
                    },
                )
            )
            .mappings()
            .all()
        )

    return {
        "object": "list",
        "data": [
            {
                "day": r["day"].isoformat(),
                "model": r["model_resolved"],
                "prompt_version": r["prompt_version"],
                "requests": r["requests"],
                "successful": r["successful"],
                "tokens": {
                    "uncached_input": r["uncached_input_tokens"] or 0,
                    "cached_input": r["cached_input_tokens"] or 0,
                    "cache_write": r["cache_write_tokens"] or 0,
                    "output": r["output_tokens"] or 0,
                    "thinking": r["thinking_tokens"] or 0,
                },
                "cost_usd": str(r["cost_usd"] or 0),
                "cost_per_successful_task": (
                    str(r["cost_per_successful_task"])
                    if r["cost_per_successful_task"] is not None
                    else None
                ),
            }
            for r in rows
        ],
    }


@router.get("/budget")
async def budget_status(principal: PrincipalDep) -> dict[str, Any]:
    from ..governance import budgets

    async with session_scope() as session:
        decision = await budgets.check(session, principal.tenant_id)
        events = []
        if principal.tenant_id is not None:
            events = [
                {
                    "created_at": r["created_at"].isoformat(),
                    "event": r["event"],
                    "spent_usd": str(r["spent_usd"]),
                }
                for r in (
                    await session.execute(
                        text("""
                            SELECT created_at, event, spent_usd FROM budget_events
                            WHERE tenant_id = :t ORDER BY created_at DESC LIMIT 20
                        """),
                        {"t": str(principal.tenant_id)},
                    )
                )
                .mappings()
                .all()
            ]
    return decision.as_json() | {"events": events}


# --- evaluation -----------------------------------------------------------


@router.get("/eval/runs")
async def eval_runs(
    principal: PrincipalDep, limit: int = Query(25, ge=1, le=200)
) -> dict[str, Any]:
    """Eval history. Every metric carries its interval, never a bare point."""
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    text("""
                    SELECT run_id, created_at, dataset, dataset_size, candidate_name,
                           baseline_name, metrics, deltas, judge_summary, judge_delta,
                           calibration, cost_usd, cost_per_success, failures,
                           regressed, tolerance, git_sha
                    FROM eval_runs ORDER BY created_at DESC LIMIT :limit
                """),
                    {"limit": limit},
                )
            )
            .mappings()
            .all()
        )

    return {
        "object": "list",
        "data": [
            {
                "run_id": r["run_id"],
                "created_at": r["created_at"].isoformat(),
                "dataset": r["dataset"],
                "dataset_size": r["dataset_size"],
                "candidate": r["candidate_name"],
                "baseline": r["baseline_name"],
                "metrics": r["metrics"],
                "deltas": r["deltas"],
                "judge": r["judge_summary"],
                "judge_delta": r["judge_delta"],
                "calibration": r["calibration"],
                "cost_usd": str(r["cost_usd"] or 0),
                "cost_per_success": (str(r["cost_per_success"]) if r["cost_per_success"] else None),
                "failures": r["failures"],
                "regressed": r["regressed"],
                "tolerance": r["tolerance"],
                "git_sha": r["git_sha"],
            }
            for r in rows
        ],
    }


@router.get("/eval/runs/{run_id}")
async def eval_run_detail(run_id: str, principal: PrincipalDep) -> dict[str, Any]:
    async with session_scope() as session:
        results = (
            (
                await session.execute(
                    text("""
                    SELECT example_id, arm, response_text, model, cost_usd, latency_ms,
                           trace_id, error, scores, judge_verdict, judge_reason
                    FROM eval_results WHERE run_id = :r ORDER BY example_id, arm
                """),
                    {"r": run_id},
                )
            )
            .mappings()
            .all()
        )

    if not results:
        raise PrismError(ErrorKind.INVALID_REQUEST, "Run not found.", status_code=404)

    return {
        "run_id": run_id,
        "results": [
            {
                "example_id": r["example_id"],
                "arm": r["arm"],
                "response": r["response_text"],
                "model": r["model"],
                "cost_usd": str(r["cost_usd"] or 0),
                "latency_ms": r["latency_ms"],
                # The join back to the gateway trace that produced this answer,
                # so a surprising score opens in the trace explorer.
                "trace_id": r["trace_id"],
                "error": r["error"],
                "scores": r["scores"],
                "judge_verdict": r["judge_verdict"],
                "judge_reason": r["judge_reason"],
            }
            for r in results
        ],
    }


# --- runtime state --------------------------------------------------------


@router.get("/health/providers")
async def provider_health(principal: PrincipalDep, chain: ChainDep) -> dict[str, Any]:
    """Circuit-breaker state per provider.

    `ignored_failures` is deliberately visible: it counts the 429s that were
    correctly *not* treated as provider ill-health, which is the distinction the
    whole error taxonomy exists to make.
    """
    if chain is None:
        return {"object": "list", "data": []}
    return {"object": "list", "data": chain.health()}


@router.get("/health/rate-limits")
async def rate_limit_health(principal: PrincipalDep, limiter: LimiterDep) -> dict[str, Any]:
    if limiter is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "limits": {
            "requests_per_minute": limiter.limits.requests_per_minute,
            "input_tokens_per_minute": limiter.limits.input_tokens_per_minute,
            "output_tokens_per_minute": limiter.limits.output_tokens_per_minute,
        },
        # Reservation accuracy, which is the metric that says whether the
        # cache-behaviour prediction is any good.
        "reconciliation": limiter.stats.as_json(),
        "effective_input_capacity": {
            f"hit_rate_{int(r * 100)}": limiter.effective_input_capacity(r)
            for r in (0.0, 0.5, 0.8, 0.9)
        },
    }


@router.get("/router")
async def router_info(principal: PrincipalDep, router_policy: RouterDep) -> dict[str, Any]:
    """The router's coefficients and the thresholds it is compared against.

    The break-even table is included because it is *derived* from the price list
    rather than tuned, and seeing the two side by side is the whole argument.
    """
    from ..routing.economics import break_even_table

    payload: dict[str, Any] = {
        "enabled": router_policy is not None,
        "break_even": break_even_table(),
    }
    if router_policy is not None:
        model = router_policy.router
        payload["trained"] = model is not None and model.is_trained
        payload["mode"] = str(router_policy.mode)
        payload["ladder"] = router_policy.ladder
        if model is not None and model.is_trained:
            weights = model.coefficients()
            payload["coefficients"] = dict(sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:15])
    return payload


@router.get("/prompts")
async def prompt_registry(principal: PrincipalDep) -> dict[str, Any]:
    """Versions, hashes, and what is pending the gate.

    Content hashes are exposed because a version bump is a cache-key change: the
    dashboard's prompt diff view uses them to show the predicted prefix-cache
    invalidation before anyone promotes anything.
    """
    from ..config import get_settings
    from ..prompts import PromptError, Registry

    registry = Registry(get_settings().prompts_root)
    data = []
    for name in registry.names():
        try:
            active = str(registry.active(name))
        except PromptError:
            active = None
        versions = []
        for version in registry.versions(name):
            prompt = registry.get(name, version)
            versions.append(
                {
                    "version": str(version),
                    "content_hash": prompt.content_hash,
                    "chars": len(prompt.text),
                    "active": str(version) == active,
                }
            )
        data.append({"name": name, "active": active, "versions": versions})
    return {"object": "list", "data": data}
