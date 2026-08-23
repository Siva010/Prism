"""Deciding what a request is allowed to cache.

The breakpoint placer takes scopes on trust. This module is what earns that
trust: it only claims `SHARED` for content it can *prove* is tenant-independent,
by checking that the bytes about to be sent are byte-identical to a versioned
artifact in the prompt registry.

That check is the whole point. "The system prompt is probably the same for
everyone" is an assumption; "these bytes equal `assistant@v1` in the registry" is
a fact. Under a shared API key the difference between the two is a cross-tenant
data leak, so the fact is what the policy runs on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..prompts import PromptError, Registry
from .breakpoints import CachePolicy, PlacementReport, Scope, apply


@dataclass(frozen=True)
class ScopeDecision:
    system: Scope
    tools: Scope
    conversation: Scope
    reason: str

    def as_json(self) -> dict[str, str]:
        return {
            "system": str(self.system),
            "tools": str(self.tools),
            "conversation": str(self.conversation),
            "reason": self.reason,
        }


def _system_text(body: dict[str, Any]) -> str:
    system = body.get("system")
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return "".join(b.get("text", "") for b in system if b.get("type") == "text")


def decide_scopes(
    body: dict[str, Any],
    *,
    prompt_version: str | None,
    registry: Registry | None,
    trust_tools: bool = True,
) -> ScopeDecision:
    """Work out what may be cached across tenants.

    `trust_tools` defaults to True because tool schemas are application-level
    artifacts checked into the repo, not per-tenant content. A deployment that
    generates tool descriptions from customer data must set it False — and that
    is a deployment decision, not something this function can detect.
    """
    conversation = Scope.TENANT  # message history is per-conversation, always
    tools = Scope.SHARED if trust_tools else Scope.TENANT

    system_text = _system_text(body)
    if not system_text:
        return ScopeDecision(Scope.TENANT, tools, conversation, "no system prompt")

    if not prompt_version or registry is None:
        return ScopeDecision(
            Scope.TENANT,
            tools,
            conversation,
            "system prompt is not registry-versioned; assumed tenant-specific",
        )

    try:
        prompt = registry.resolve(prompt_version)
    except PromptError as exc:
        return ScopeDecision(
            Scope.TENANT, tools, conversation, f"prompt_version did not resolve: {exc}"
        )

    if prompt.text != system_text:
        # The claimed version exists but the bytes differ, so the caller
        # assembled something per-request on top of it. Caching the result under
        # a shared key would leak whatever they added.
        return ScopeDecision(
            Scope.TENANT,
            tools,
            conversation,
            f"system prompt does not match {prompt.ref} byte-for-byte "
            "(caller modified it per request)",
        )

    return ScopeDecision(
        Scope.SHARED,
        tools,
        conversation,
        f"system prompt is byte-identical to {prompt.ref} ({prompt.content_hash})",
    )


def apply_to_request(
    body: dict[str, Any],
    policy: CachePolicy,
    *,
    prompt_version: str | None,
    registry: Registry | None,
    trust_tools: bool = True,
) -> tuple[dict[str, Any], PlacementReport, ScopeDecision]:
    decision = decide_scopes(
        body,
        prompt_version=prompt_version,
        registry=registry,
        trust_tools=trust_tools,
    )
    marked, report = apply(
        body,
        policy,
        system_scope=decision.system,
        tools_scope=decision.tools,
        conversation_prefix_scope=decision.conversation,
    )
    return marked, report, decision
