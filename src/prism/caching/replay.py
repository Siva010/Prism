"""The three-configuration replay: no cache, prefix only, prefix + semantic.

The headline result this project exists to produce is **not** "the semantic cache
has an X% hit rate". It is the *incremental* cost delta of `prefix + semantic`
over `prefix alone`, together with the false-hit exposure that third
configuration adds. Prefix caching already captures the easy wins at zero
correctness risk, so the semantic layer has to justify itself on the residual.

**This is a simulation, and says so.** Replaying a corpus against the live API
three times would cost three times as much and still not isolate the variable,
because the provider's cache state is not controllable from outside. So the
simulation replays recorded requests through the real breakpoint policy, the real
scope keys, the real embedder, and the real five-class cost model, and models
only two things: whether a prefix was live in the provider's cache at that moment
(a TTL window over previously-seen prefixes), and whether a semantic lookup would
have hit.

What that buys is a controlled comparison of three configurations over one fixed
corpus. What it does not buy is a claim about production. The distinction belongs
in any report built from it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from ..cost import CostBreakdown, TokenUsage, compute_cost
from ..prompts import Registry
from ..registry import ModelSpec, resolve_model
from .breakpoints import CachePolicy
from .breakpoints import apply as place_breakpoints
from .embeddings import Embedder
from .keys import CacheScope, query_text
from .policy import decide_scopes
from .semantic import CacheHit, SemanticCache, SemanticCacheConfig
from .store import InMemoryStore

TTL_SECONDS = {"5m": 300, "1h": 3600}


@dataclass(frozen=True)
class ReplayRequest:
    """One recorded request/response pair from the trace table."""

    request_id: str
    tenant_id: str
    timestamp: datetime
    body: dict[str, Any]
    response: dict[str, Any]
    usage: TokenUsage
    prompt_version: str | None = None

    @property
    def model(self) -> str:
        return str(self.body.get("model", ""))


@dataclass
class ConfigResult:
    name: str
    requests: int = 0
    upstream_calls: int = 0
    semantic_hits: int = 0
    prefix_reads: int = 0
    prefix_writes: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: Decimal = Decimal(0)
    breakdown: list[CostBreakdown] = field(default_factory=list)
    # Every served-from-cache decision, so exposure can be audited afterwards
    # rather than trusted.
    served: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def semantic_hit_rate(self) -> float:
        return self.semantic_hits / self.requests if self.requests else 0.0

    @property
    def cost_per_request(self) -> Decimal:
        return self.cost / self.requests if self.requests else Decimal(0)

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requests": self.requests,
            "upstream_calls": self.upstream_calls,
            "semantic_hits": self.semantic_hits,
            "semantic_hit_rate": self.semantic_hit_rate,
            "prefix_read_tokens": self.usage.cache_read_input_tokens,
            "prefix_write_tokens": self.usage.cache_creation_input_tokens,
            "uncached_input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cost_usd": str(self.cost),
            "cost_per_request_usd": str(self.cost_per_request),
        }


@dataclass
class ReplayReport:
    no_cache: ConfigResult
    prefix_only: ConfigResult
    prefix_and_semantic: ConfigResult
    false_hit_rate: float | None = None
    false_hit_upper_bound: float | None = None
    threshold: float = 0.0
    embedder: str = ""

    def _saving(self, better: ConfigResult, worse: ConfigResult) -> float:
        if worse.cost == 0:
            return 0.0
        return float((worse.cost - better.cost) / worse.cost)

    @property
    def prefix_saving(self) -> float:
        """Prefix caching against no caching. Free, in correctness terms."""
        return self._saving(self.prefix_only, self.no_cache)

    @property
    def incremental_saving(self) -> float:
        """The number that matters: semantic *on top of* prefix caching."""
        return self._saving(self.prefix_and_semantic, self.prefix_only)

    @property
    def total_saving(self) -> float:
        return self._saving(self.prefix_and_semantic, self.no_cache)

    def as_json(self) -> dict[str, Any]:
        return {
            "simulation": True,
            "configurations": [
                self.no_cache.as_json(),
                self.prefix_only.as_json(),
                self.prefix_and_semantic.as_json(),
            ],
            "prefix_saving_vs_no_cache": self.prefix_saving,
            "incremental_saving_vs_prefix": self.incremental_saving,
            "total_saving_vs_no_cache": self.total_saving,
            "false_hit_rate": self.false_hit_rate,
            "false_hit_upper_bound": self.false_hit_upper_bound,
            "threshold": self.threshold,
            "embedder": self.embedder,
        }

    def render(self) -> str:
        rows = [self.no_cache, self.prefix_only, self.prefix_and_semantic]
        lines = [
            f"replay over {self.no_cache.requests} requests "
            f"(SIMULATED cache state, not a live measurement)",
            "",
            f"  {'configuration':<22} {'calls':>7} {'cost':>12} {'per req':>12}",
        ]
        for row in rows:
            lines.append(
                f"  {row.name:<22} {row.upstream_calls:>7} "
                f"{'$' + format(row.cost, '.4f'):>12} "
                f"{'$' + format(row.cost_per_request, '.6f'):>12}"
            )
        lines += [
            "",
            f"  prefix caching vs none:        {self.prefix_saving:+.1%}   (no correctness risk)",
            f"  semantic ON TOP of prefix:     {self.incremental_saving:+.1%}   "
            "<- the number worth reporting",
            f"  both vs none:                  {self.total_saving:+.1%}",
        ]
        if self.false_hit_rate is not None:
            lines += [
                "",
                f"  cost of that increment: false-hit rate {self.false_hit_rate:.2%} "
                f"(95% CI upper bound {self.false_hit_upper_bound:.2%}) "
                f"at threshold {self.threshold:.3f}",
            ]
            if self.incremental_saving < 0.05:
                lines += [
                    "",
                    "  The incremental saving is small. Reporting that honestly is a",
                    "  stronger result than a headline hit rate: prefix caching already",
                    "  captured the easy wins at zero correctness risk, and this layer",
                    "  buys little on the residual while adding false-hit exposure.",
                ]
        return "\n".join(lines)


class _PrefixSimulator:
    """Models whether a marked prefix was live in the provider's cache.

    Keyed on the exact prefix content (scope key plus the placement's token
    boundary), with a TTL window. First sight of a prefix is a write; a sighting
    inside the window is a read; outside it, a write again.
    """

    def __init__(self, ttl: str = "5m") -> None:
        self.window = timedelta(seconds=TTL_SECONDS.get(ttl, 300))
        self._last_seen: dict[str, datetime] = {}

    def observe(self, key: str, at: datetime) -> bool:
        """True if this counts as a cache read rather than a write."""
        previous = self._last_seen.get(key)
        self._last_seen[key] = at
        return previous is not None and (at - previous) <= self.window


async def replay(
    corpus: list[ReplayRequest],
    embedder: Embedder,
    *,
    threshold: float,
    ttl: str = "5m",
    prompts_root: str = "prompts",
    min_query_chars: int = 24,
) -> ReplayReport:
    """Run the corpus through all three configurations."""
    report = ReplayReport(
        no_cache=ConfigResult("no cache"),
        prefix_only=ConfigResult("prefix only"),
        prefix_and_semantic=ConfigResult("prefix + semantic"),
        threshold=threshold,
        embedder=embedder.model_name,
    )
    registry = Registry(prompts_root)

    semantic = SemanticCache(
        InMemoryStore(),
        embedder,
        SemanticCacheConfig(enabled=True, threshold=threshold, min_query_chars=min_query_chars),
        allow_unsafe_embedder=not embedder.is_production_grade,
    )
    prefix_sim = _PrefixSimulator(ttl)
    prefix_sim_semantic = _PrefixSimulator(ttl)

    for entry in corpus:
        spec = resolve_model(entry.model)

        _accumulate_no_cache(report.no_cache, entry, spec)

        report.prefix_only.requests += 1
        _accumulate_prefix(report.prefix_only, entry, spec, prefix_sim, registry, ttl)

        config = report.prefix_and_semantic
        config.requests += 1
        scope = CacheScope.from_request(
            entry.body, tenant_id=entry.tenant_id, prompt_version=entry.prompt_version
        )
        result = await semantic.lookup(entry.body, scope)
        if isinstance(result, CacheHit):
            # Served from cache: no upstream call, so no tokens and no cost.
            config.semantic_hits += 1
            config.served.append((entry.request_id, result.matched_query, result.similarity))
            continue

        _accumulate_prefix(config, entry, spec, prefix_sim_semantic, registry, ttl)
        await semantic.put(
            entry.body,
            scope,
            entry.response,
            cost_usd=float(compute_cost(entry.usage, spec).total_usd),
        )

    return report


def _accumulate_no_cache(config: ConfigResult, entry: ReplayRequest, spec: ModelSpec) -> None:
    config.requests += 1
    config.upstream_calls += 1
    # Every input token at full price: no reads, no writes.
    usage = TokenUsage(
        input_tokens=entry.usage.billable_input_tokens,
        output_tokens=entry.usage.output_tokens,
    )
    cost = compute_cost(usage, spec)
    config.usage = _add(config.usage, usage)
    config.cost += cost.total_usd


def _accumulate_prefix(
    config: ConfigResult,
    entry: ReplayRequest,
    spec: ModelSpec,
    simulator: _PrefixSimulator,
    registry: Registry,
    ttl: str,
) -> None:
    config.upstream_calls += 1

    decision = decide_scopes(entry.body, prompt_version=entry.prompt_version, registry=registry)
    policy = CachePolicy.for_tier(spec.tier, ttl=ttl)
    _, placement = place_breakpoints(
        entry.body,
        policy,
        system_scope=decision.system,
        tools_scope=decision.tools,
        conversation_prefix_scope=decision.conversation,
    )

    total_input = entry.usage.billable_input_tokens
    if not placement.placements:
        usage = TokenUsage(input_tokens=total_input, output_tokens=entry.usage.output_tokens)
    else:
        cacheable = min(placement.estimated_cacheable_tokens, total_input)
        scope = CacheScope.from_request(
            entry.body, tenant_id=entry.tenant_id, prompt_version=entry.prompt_version
        )
        prefix_key = f"{scope.key}:{placement.placements[-1].prefix_tokens}"
        is_read = simulator.observe(prefix_key, entry.timestamp)
        if is_read:
            config.prefix_reads += 1
            usage = TokenUsage(
                input_tokens=total_input - cacheable,
                cache_read_input_tokens=cacheable,
                output_tokens=entry.usage.output_tokens,
            )
        else:
            config.prefix_writes += 1
            usage = TokenUsage(
                input_tokens=total_input - cacheable,
                cache_creation_input_tokens=cacheable,
                output_tokens=entry.usage.output_tokens,
            )

    cost = compute_cost(usage, spec, cache_ttl=ttl)
    config.usage = _add(config.usage, usage)
    config.cost += cost.total_usd


def _add(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        cache_read_input_tokens=a.cache_read_input_tokens + b.cache_read_input_tokens,
        cache_creation_input_tokens=a.cache_creation_input_tokens + b.cache_creation_input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        thinking_tokens=a.thinking_tokens + b.thinking_tokens,
    )


def audit_served(report: ReplayReport, corpus: list[ReplayRequest]) -> list[dict[str, Any]]:
    """Every semantic hit, paired with what it was actually asked.

    A hit rate is only trustworthy if the hits can be inspected. This is what a
    human reads to label false hits, and what turns the replay's exposure figure
    into a measured one rather than an assumed one.
    """
    by_id = {r.request_id: r for r in corpus}
    return [
        {
            "request_id": request_id,
            "asked": query_text(by_id[request_id].body) if request_id in by_id else "",
            "served_answer_to": matched,
            "similarity": similarity,
            "false_hit": None,  # fill in: true | false
        }
        for request_id, matched, similarity in report.prefix_and_semantic.served
    ]


def run(corpus: list[ReplayRequest], embedder: Embedder, **kwargs: Any) -> ReplayReport:
    return asyncio.run(replay(corpus, embedder, **kwargs))
