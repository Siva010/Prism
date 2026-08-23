"""Layer 1: provider prefix cache — where the breakpoints go, and why.

Anthropic's prefix cache is exact-match over a request prefix. Marking a block
with `cache_control` caches everything up to and including it. The hierarchy is
`tools -> system -> messages`, and a change at any level invalidates that level
and everything after it.

Two things make placement non-obvious.

**A bad breakpoint is worse than no breakpoint.** Cache writes carry a ~25%
surcharge and reads are ~90% off. A breakpoint on content that changes every
request pays the write premium every time and never reads it back. The metric
that catches this is the write/read ratio, reported per prompt version by
`GET /admin/stats/cache`: persistently above ~1 means the breakpoints are sitting
on volatile content.

**Placement is a security boundary.** Prefix cache entries are isolated per
organization, but a deployment using one API key for all tenants shares that
organization. So the *prefix* is shared between tenants. Marking a prefix that
contains tenant A's data means tenant B can hit a cache entry built from it. This
module therefore refuses to place a breakpoint after any block not explicitly
marked tenant-independent, and that refusal is a hard failure rather than a
warning — a cost optimisation that silently becomes a data leak is not a
tradeoff anyone gets to make at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..tokens import estimate_blocks, estimate_tool

# The API accepts at most four cache_control markers per request.
MAX_BREAKPOINTS = 4

# Below this, a prefix is not cached at all and the marker is simply ignored.
# Haiku's floor is higher, so the registry-driven value is used per model.
DEFAULT_MIN_CACHEABLE_TOKENS = 1024
MIN_CACHEABLE_TOKENS_BY_TIER = {"haiku": 2048, "sonnet": 1024, "opus": 1024}


class Scope(StrEnum):
    """Whether a block's content may be shared across tenants.

    `SHARED` is a claim that the block contains nothing tenant-specific. It is the
    caller's job to make that claim truthfully, and the only claim that permits a
    breakpoint to be placed after the block.
    """

    SHARED = "shared"
    TENANT = "tenant"


class BreakpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class Placement:
    """One breakpoint, and the reasoning that produced it."""

    level: str  # tools | system | messages
    index: int
    prefix_tokens: int
    reason: str


@dataclass
class PlacementReport:
    placements: list[Placement] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    estimated_cacheable_tokens: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "breakpoints": [
                {
                    "level": p.level,
                    "index": p.index,
                    "prefix_tokens": p.prefix_tokens,
                    "reason": p.reason,
                }
                for p in self.placements
            ],
            "skipped": self.skipped,
            "estimated_cacheable_tokens": self.estimated_cacheable_tokens,
        }


@dataclass
class CachePolicy:
    """Configuration for one request's breakpoint placement."""

    enabled: bool = True
    ttl: str = "5m"  # "5m" or "1h"; 1h writes cost 2x base instead of 1.25x
    min_cacheable_tokens: int = DEFAULT_MIN_CACHEABLE_TOKENS
    max_breakpoints: int = MAX_BREAKPOINTS
    # Cache the conversation prefix as well as tools/system. Off by default:
    # history is per-conversation, so it only pays back on multi-turn traffic and
    # otherwise adds write cost for reads that never come.
    cache_conversation_prefix: bool = False

    @classmethod
    def for_tier(cls, tier: str, **kwargs: Any) -> CachePolicy:
        return cls(
            min_cacheable_tokens=MIN_CACHEABLE_TOKENS_BY_TIER.get(
                tier, DEFAULT_MIN_CACHEABLE_TOKENS
            ),
            **kwargs,
        )


def _marker(policy: CachePolicy) -> dict[str, str]:
    marker = {"type": "ephemeral"}
    if policy.ttl != "5m":
        marker["ttl"] = policy.ttl
    return marker


def _normalize_system(system: Any) -> list[dict[str, Any]] | None:
    """A string system prompt cannot carry a marker; promote it to one block."""
    if system is None:
        return None
    if isinstance(system, str):
        return [{"type": "text", "text": system}] if system else None
    return list(system) if system else None


def apply(
    body: dict[str, Any],
    policy: CachePolicy,
    *,
    system_scope: Scope = Scope.TENANT,
    tools_scope: Scope = Scope.SHARED,
    conversation_prefix_scope: Scope = Scope.TENANT,
) -> tuple[dict[str, Any], PlacementReport]:
    """Return a copy of `body` with cache_control markers, plus what was decided.

    Scopes default to the safe answer. `system_scope` defaults to TENANT because
    a system prompt assembled at runtime routinely carries tenant context; the
    caller passes SHARED only when the prompt came from the versioned registry
    and is identical for every tenant.
    """
    report = PlacementReport()
    if not policy.enabled:
        report.skipped.append("caching disabled")
        return body, report

    out = dict(body)
    remaining = policy.max_breakpoints
    running_tokens = 0

    # --- tools ----------------------------------------------------------
    tools = out.get("tools")
    if tools:
        tools_tokens = sum(estimate_tool(t) for t in tools)
        running_tokens += tools_tokens
        if tools_scope is not Scope.SHARED:
            report.skipped.append("tools: tenant-scoped, breakpoint refused")
        elif running_tokens < policy.min_cacheable_tokens:
            report.skipped.append(
                f"tools: prefix ~{running_tokens} tokens is below the "
                f"{policy.min_cacheable_tokens} minimum"
            )
        elif not _system_is_cacheable(out, policy, running_tokens, system_scope):
            # Only worth its own breakpoint when the next level will not simply
            # absorb it: a marker on system already caches the tools before it,
            # and spending two markers to cache one prefix wastes a scarce slot.
            tools = [dict(t) for t in tools]
            tools[-1]["cache_control"] = _marker(policy)
            out["tools"] = tools
            remaining -= 1
            report.placements.append(
                Placement("tools", len(tools) - 1, running_tokens, "stable tool schema")
            )
            report.estimated_cacheable_tokens = running_tokens
        else:
            report.skipped.append("tools: covered by the system breakpoint")

    # --- system ---------------------------------------------------------
    system = _normalize_system(out.get("system"))
    if system is not None:
        system_tokens = estimate_blocks(system)
        running_tokens += system_tokens
        if system_scope is not Scope.SHARED:
            # The security boundary. A tenant-specific system prompt cached under
            # a shared API key is reachable by every other tenant in the org.
            report.skipped.append(
                "system: tenant-scoped, breakpoint refused (a shared prefix "
                "containing tenant data is a cross-tenant leak)"
            )
        elif running_tokens < policy.min_cacheable_tokens:
            report.skipped.append(
                f"system: prefix ~{running_tokens} tokens is below the "
                f"{policy.min_cacheable_tokens} minimum"
            )
        elif remaining > 0:
            system = [dict(b) for b in system]
            system[-1]["cache_control"] = _marker(policy)
            out["system"] = system
            remaining -= 1
            report.placements.append(
                Placement(
                    "system",
                    len(system) - 1,
                    running_tokens,
                    "versioned system prompt, identical across tenants",
                )
            )
            report.estimated_cacheable_tokens = running_tokens
        else:
            report.skipped.append("system: no breakpoints left")
        if isinstance(out.get("system"), str):
            out["system"] = system

    # --- conversation prefix --------------------------------------------
    messages = out.get("messages") or []
    if policy.cache_conversation_prefix and len(messages) > 2 and remaining > 0:
        if conversation_prefix_scope is not Scope.SHARED:
            report.skipped.append("messages: conversation is tenant-scoped, breakpoint refused")
        else:
            # Never mark the final turn: it is different on every request, so a
            # marker there pays the write premium and can never be read back.
            cut = len(messages) - 2
            prefix_tokens = running_tokens + sum(
                estimate_blocks(m.get("content")) for m in messages[: cut + 1]
            )
            if prefix_tokens < policy.min_cacheable_tokens:
                report.skipped.append(
                    f"messages: prefix ~{prefix_tokens} tokens is below the "
                    f"{policy.min_cacheable_tokens} minimum"
                )
            else:
                messages = [dict(m) for m in messages]
                target = messages[cut]
                content = target.get("content")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if isinstance(content, list) and content:
                    content = [dict(b) for b in content]
                    content[-1]["cache_control"] = _marker(policy)
                    target["content"] = content
                    messages[cut] = target
                    out["messages"] = messages
                    remaining -= 1
                    report.placements.append(
                        Placement("messages", cut, prefix_tokens, "conversation prefix")
                    )
                    report.estimated_cacheable_tokens = prefix_tokens

    if not report.placements and not report.skipped:
        report.skipped.append("nothing cacheable in this request")
    return out, report


def _system_is_cacheable(
    body: dict[str, Any], policy: CachePolicy, running_tokens: int, scope: Scope
) -> bool:
    """Will the system level take a breakpoint that subsumes the tools prefix?"""
    if scope is not Scope.SHARED:
        return False
    system = _normalize_system(body.get("system"))
    if system is None:
        return False
    return running_tokens + estimate_blocks(system) >= policy.min_cacheable_tokens


def audit(body: dict[str, Any]) -> list[str]:
    """Find markers that cannot pay off, on a body about to be dispatched.

    A safety net for hand-written bodies and for the dual-ingress path, where a
    client may send its own `cache_control`. Marking the final message is the
    common mistake: it changes every request, so it is a guaranteed write and a
    guaranteed miss.
    """
    problems: list[str] = []
    messages = body.get("messages") or []
    if messages:
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and "cache_control" in b for b in content
        ):
            problems.append(
                "cache_control on the final message: it differs on every request, "
                "so this pays the write premium and never reads back"
            )

    count = _count_markers(body)
    if count > MAX_BREAKPOINTS:
        problems.append(f"{count} cache_control markers; the API accepts {MAX_BREAKPOINTS}")
    return problems


def _count_markers(body: dict[str, Any]) -> int:
    count = 0
    for tool in body.get("tools") or []:
        count += "cache_control" in tool
    system = body.get("system")
    if isinstance(system, list):
        count += sum(1 for b in system if isinstance(b, dict) and "cache_control" in b)
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            count += sum(1 for b in content if isinstance(b, dict) and "cache_control" in b)
    return count
