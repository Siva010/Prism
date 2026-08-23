"""Concrete arms and judge clients.

**The arm calls Prism over HTTP, not the translation layer directly.** An eval
that bypasses the gateway measures the model; the point here is to measure the
*system*, including routing, caching, and any bug the gateway itself introduces.
Every eval response also leaves a trace, so a surprising score can be traced back
to the exact request that produced it.

**The judge does not go through Prism.** It is a client of a model, not a proxy
for one, and routing it through the system under test would make the measuring
instrument depend on the thing being measured.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import httpx

from .dataset import Example
from .runner import Arm, Response


class JudgeConfigurationError(RuntimeError):
    pass


# Model-name fragments that indicate the Anthropic family. Prism serves Claude on
# every tier, so a judge matching any of these would be scoring its own family.
_CLAUDE_MARKERS = ("claude", "anthropic", "opus", "sonnet", "haiku")


def assert_different_family(judge_model: str) -> None:
    """Refuse a judge drawn from the system under test's own family.

    Models rate their own outputs higher. Since Prism serves Claude across the
    whole ladder, a Claude judge would inflate every score in a way no amount of
    position-swapping or bootstrapping can undo. Staying single-vendor on the
    serving side is what makes this checkable rather than a rule someone has to
    remember.
    """
    lowered = judge_model.lower()
    if any(marker in lowered for marker in _CLAUDE_MARKERS):
        raise JudgeConfigurationError(
            f"judge model {judge_model!r} appears to be from the same family as the "
            "system under test. Self-preference bias would inflate every score. "
            "Configure PRISM_JUDGE_MODEL with a non-Anthropic model."
        )


class OpenAICompatibleJudge:
    """Judge client speaking the OpenAI Chat Completions API.

    Works against OpenAI, or any compatible endpoint — the secondary provider
    that weeks 10-11 add for failover doubles as the judge, which is why the
    spec keeps a second vendor even though one upstream would serve traffic.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 120.0,
        max_tokens: int = 512,
    ) -> None:
        assert_different_family(model)
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers={"authorization": f"Bearer {api_key}"},
        )

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                # The judge is a measuring instrument; a deterministic setting
                # means re-running a calibration set reproduces its own numbers.
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"] or "")

    async def aclose(self) -> None:
        await self._client.aclose()


class GatewayArm:
    """An arm that answers examples by calling a running Prism instance."""

    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        prompt_version: str | None = None,
        system_prompt: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self.name = name
        self.model = model
        self.prompt_version = prompt_version
        self.system_prompt = system_prompt
        self.extra_body = extra_body or {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers={"authorization": f"Bearer {api_key}"},
        )

    def as_arm(self) -> Arm:
        return Arm(
            name=self.name,
            run=self.answer,
            config={
                "model": self.model,
                "prompt_version": self.prompt_version,
                "system_prompt": self.system_prompt,
                **self.extra_body,
            },
        )

    async def answer(self, example: Example) -> Response:
        messages = list(example.messages)
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}, *messages]

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.extra_body,
        }
        if self.prompt_version:
            body["prism"] = {
                **body.get("prism", {}),
                "prompt_version": self.prompt_version,
            }
        if example.schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": example.id, "schema": example.schema},
            }

        started = time.perf_counter()
        try:
            http = await self._client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            return Response(example.id, "", error=f"transport: {exc}")

        latency_ms = int((time.perf_counter() - started) * 1000)
        if http.status_code >= 400:
            return Response(
                example.id,
                "",
                latency_ms=latency_ms,
                error=f"http {http.status_code}: {http.text[:200]}",
            )

        payload = http.json()
        text = payload["choices"][0]["message"].get("content") or ""
        return Response(
            example_id=example.id,
            text=text,
            model=http.headers.get("x-prism-model", self.model),
            # Read from the gateway's own accounting rather than recomputed here,
            # so the eval's cost figures and the trace table cannot disagree.
            cost_usd=Decimal(http.headers.get("x-prism-cost-usd", "0")),
            latency_ms=latency_ms,
            trace_id=http.headers.get("x-request-id"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
