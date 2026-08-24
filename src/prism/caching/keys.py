"""Cache key scoping.

A semantic cache lookup is a nearest-neighbour search, so the *scope* of that
search is what decides whether a near-miss is merely unhelpful or actively wrong.
Everything that changes what a correct answer looks like has to be part of the
scope, not part of the similarity:

* **tenant** — the one that matters. If tenant A's cached response can reach
  tenant B, that is a data leak, and no similarity threshold prevents it. Scoped,
  never compared. There is a test that proves it.
* **model** — Haiku and Opus give different answers to the same question. A hit
  across tiers would silently serve the cheap answer to someone who paid for the
  expensive one, and would also destroy the router's measurements in week 9.
* **temperature** — a request that asked for varied output does not want a
  replayed one.
* **system prompt hash** — the same question under a different system prompt is a
  different question. This is also why prompt versions are cache-key inputs: a
  version bump has a predictable, measurable cache cost rather than a mysterious
  hit-rate cliff.
* **tool schema hash** — a model that can call tools answers differently from one
  that cannot.

Scope is enforced by *partitioning* rather than by filtering after retrieval.
Retrieving across scopes and discarding the wrong ones would mean the leak exists
in the query path and is prevented only by a later `if`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _hash(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        payload = value
    else:
        # sort_keys, because an unsorted json.dumps makes an identical tool set
        # hash differently between runs and silently empties the cache.
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CacheScope:
    """Everything that must match exactly before similarity is even considered."""

    tenant_id: str
    model: str
    system_prompt_hash: str = "none"
    tool_schema_hash: str = "none"
    temperature: float | None = None
    prompt_version: str | None = None

    @classmethod
    def from_request(
        cls,
        body: dict[str, Any],
        *,
        tenant_id: str,
        prompt_version: str | None = None,
    ) -> CacheScope:
        system = body.get("system")
        if isinstance(system, list):
            system_text = "".join(b.get("text", "") for b in system if b.get("type") == "text")
        else:
            system_text = system or ""

        tools = body.get("tools") or []
        tool_signature = [
            {"name": t.get("name"), "input_schema": t.get("input_schema")} for t in tools
        ]

        return cls(
            tenant_id=tenant_id,
            model=body.get("model", ""),
            system_prompt_hash=_hash(system_text) if system_text else "none",
            tool_schema_hash=_hash(tool_signature) if tools else "none",
            temperature=body.get("temperature"),
            prompt_version=prompt_version,
        )

    @property
    def key(self) -> str:
        """One opaque partition key. Two requests may only be compared when equal."""
        return _hash(
            [
                self.tenant_id,
                self.model,
                self.system_prompt_hash,
                self.tool_schema_hash,
                self.temperature,
                self.prompt_version,
            ]
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "scope_key": self.key,
            "tenant_id": self.tenant_id,
            "model": self.model,
            "system_prompt_hash": self.system_prompt_hash,
            "tool_schema_hash": self.tool_schema_hash,
            "temperature": self.temperature,
            "prompt_version": self.prompt_version,
        }


def query_text(body: dict[str, Any]) -> str:
    """The text that gets embedded.

    Only the final user turn. Embedding the whole conversation would make every
    long thread a unique vector and drive the hit rate to zero; embedding the
    system prompt would double-count something already in the scope key.
    """
    messages = body.get("messages") or []
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if text:
                return text
    return ""


def exact_key(body: dict[str, Any], scope: CacheScope) -> str:
    """Key for the exact-match layer: the whole request, verbatim, within a scope."""
    return _hash([scope.key, json.dumps(body.get("messages"), sort_keys=True)])
