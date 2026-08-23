"""Anthropic Messages API egress.

**Why raw httpx and not the Anthropic SDK.** Prism is a gateway: by the time a
request reaches this layer it is already a fully-formed Messages API body built
by the translation layer, and what comes back has to be preserved as a body plus
its response headers. Two things the SDK abstracts away are load-bearing here —
the `anthropic-ratelimit-*` response headers that drive the token bucket
(weeks 10-11), and byte-level SSE frames for the streaming tee (week 3). The SDK
remains the right choice everywhere Prism is a *client* rather than a proxy: the
eval judge, the embedding backfills, and the batch runner.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from ..schemas.errors import ErrorKind, PrismError
from .base import ProviderResponse, RateLimitSnapshot, StreamHandle

ANTHROPIC_VERSION = "2023-06-01"


def _int_header(headers: httpx.Headers, name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _float_header(headers: httpx.Headers, name: str) -> float | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_rate_limits(headers: httpx.Headers) -> RateLimitSnapshot:
    return RateLimitSnapshot(
        requests_limit=_int_header(headers, "anthropic-ratelimit-requests-limit"),
        requests_remaining=_int_header(headers, "anthropic-ratelimit-requests-remaining"),
        requests_reset=headers.get("anthropic-ratelimit-requests-reset"),
        input_tokens_limit=_int_header(headers, "anthropic-ratelimit-input-tokens-limit"),
        input_tokens_remaining=_int_header(headers, "anthropic-ratelimit-input-tokens-remaining"),
        input_tokens_reset=headers.get("anthropic-ratelimit-input-tokens-reset"),
        output_tokens_limit=_int_header(headers, "anthropic-ratelimit-output-tokens-limit"),
        output_tokens_remaining=_int_header(headers, "anthropic-ratelimit-output-tokens-remaining"),
        output_tokens_reset=headers.get("anthropic-ratelimit-output-tokens-reset"),
        retry_after_s=_float_header(headers, "retry-after"),
    )


def classify_status(status_code: int) -> ErrorKind:
    """Map an upstream HTTP status onto Prism's error taxonomy.

    The 429/529 split is the one that matters. A 429 means *this account* is out
    of quota: back off, respect `retry-after`, and do not open the circuit —
    failing over would carry the same account's exhausted quota to another route.
    A 529 means the *provider* is over capacity: quota is irrelevant, fail over
    or wait, and this one does count toward the breaker.
    """
    if status_code == 400:
        return ErrorKind.UPSTREAM_INVALID_REQUEST
    if status_code in (401, 403):
        return ErrorKind.UPSTREAM_AUTH
    if status_code == 404:
        return ErrorKind.UPSTREAM_NOT_FOUND
    if status_code == 429:
        return ErrorKind.UPSTREAM_RATE_LIMIT
    if status_code == 529:
        return ErrorKind.UPSTREAM_OVERLOADED
    if status_code >= 500:
        return ErrorKind.UPSTREAM_SERVER_ERROR
    return ErrorKind.UPSTREAM_INVALID_REQUEST


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com",
        timeout_s: float = 600.0,
        connect_timeout_s: float = 5.0,
        max_connections: int = 200,
    ) -> None:
        self._connect_timeout_s = connect_timeout_s
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )

    async def create_message(
        self, body: dict[str, Any], *, timeout_s: float | None = None
    ) -> ProviderResponse:
        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/v1/messages",
                json=body,
                timeout=timeout_s if timeout_s is not None else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.TimeoutException as exc:
            raise PrismError(
                ErrorKind.UPSTREAM_TIMEOUT,
                f"Upstream request timed out: {exc}",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise PrismError(
                ErrorKind.UPSTREAM_CONNECTION,
                f"Could not reach the upstream provider: {exc}",
                status_code=502,
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        rate_limits = parse_rate_limits(response.headers)
        request_id = response.headers.get("request-id")

        if response.status_code >= 400:
            kind = classify_status(response.status_code)
            try:
                payload = response.json()
                message = (payload.get("error") or {}).get("message") or response.text
            except ValueError:
                payload = {"raw": response.text}
                message = response.text
            raise PrismError(
                kind,
                message,
                # A 429 is relayed as a 429 and a 529 as a 503 with retry-after
                # intact, so a well-behaved client can do the right thing without
                # knowing which upstream served it.
                status_code=429
                if kind is ErrorKind.UPSTREAM_RATE_LIMIT
                else (503 if kind is ErrorKind.UPSTREAM_OVERLOADED else response.status_code),
                retry_after=rate_limits.retry_after_s,
                upstream_status=response.status_code,
                upstream_body=payload,
            )

        return ProviderResponse(
            payload=response.json(),
            status_code=response.status_code,
            provider_request_id=request_id,
            latency_ms=latency_ms,
            rate_limits=rate_limits,
        )

    @asynccontextmanager
    async def stream_message(
        self,
        body: dict[str, Any],
        *,
        first_token_timeout_s: float | None = None,
        total_timeout_s: float | None = None,
    ) -> AsyncIterator[StreamHandle]:
        """Open an SSE stream. The context manager owns closing the connection."""
        payload = {**body, "stream": True}
        request = self._client.build_request(
            "POST",
            "/v1/messages",
            json=payload,
            headers={"accept": "text/event-stream"},
            # httpx's read timeout would fire between SSE frames, which on a
            # healthy but slow generation is a false positive. The real budgets
            # are enforced by StreamHandle against absolute deadlines.
            timeout=httpx.Timeout(total_timeout_s, connect=self._connect_timeout_s, read=None),
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise PrismError(
                ErrorKind.UPSTREAM_TIMEOUT,
                f"Upstream stream did not open in time: {exc}",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise PrismError(
                ErrorKind.UPSTREAM_CONNECTION,
                f"Could not reach the upstream provider: {exc}",
                status_code=502,
            ) from exc

        try:
            if response.status_code >= 400:
                # The failure arrived before any bytes were streamed, so it can
                # still be surfaced as a normal HTTP error with a status code.
                await response.aread()
                kind = classify_status(response.status_code)
                try:
                    error_payload = response.json()
                    message = (error_payload.get("error") or {}).get("message") or response.text
                except ValueError:
                    error_payload = {"raw": response.text}
                    message = response.text
                rate_limits = parse_rate_limits(response.headers)
                raise PrismError(
                    kind,
                    message,
                    status_code=429
                    if kind is ErrorKind.UPSTREAM_RATE_LIMIT
                    else (503 if kind is ErrorKind.UPSTREAM_OVERLOADED else response.status_code),
                    retry_after=rate_limits.retry_after_s,
                    upstream_status=response.status_code,
                    upstream_body=error_payload,
                )

            yield StreamHandle(
                response.aiter_lines(),
                provider_request_id=response.headers.get("request-id"),
                rate_limits=parse_rate_limits(response.headers),
                first_token_timeout_s=first_token_timeout_s,
                total_timeout_s=total_timeout_s,
            )
        finally:
            # Closing an unfinished response tells the upstream to stop
            # generating, which is what makes abandoning a disconnected client's
            # request actually save tokens.
            await response.aclose()

    async def count_tokens(self, body: dict[str, Any]) -> int:
        """Exact input-token count from the provider.

        The ground-truth oracle for `tokens.calibrate()`. It costs a full round
        trip, so it is never called on the hot path — the local estimator is used
        there, and this is what measures how wrong the estimator is.
        """
        payload = {
            k: v
            for k, v in body.items()
            if k in ("model", "messages", "system", "tools", "tool_choice")
        }
        response = await self._client.post("/v1/messages/count_tokens", json=payload)
        if response.status_code >= 400:
            raise PrismError(
                classify_status(response.status_code),
                f"token counting failed: {response.text[:200]}",
                status_code=response.status_code,
            )
        return int(response.json()["input_tokens"])

    async def aclose(self) -> None:
        await self._client.aclose()
