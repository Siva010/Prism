"""OpenAI-compatible ingress: `/v1/chat/completions`, streaming and not."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..cost import compute_cost
from ..logging_setup import get_logger
from ..registry import MODELS, ModelSpec, UnknownModelError, resolve_model
from ..schemas.errors import ErrorKind, PrismError
from ..schemas.openai import ChatCompletionRequest
from ..tracing.recorder import TraceDraft, TraceRecorder
from ..translation import request as request_translator
from ..translation import response as response_translator
from ..translation.mapping import to_finish_reason
from ..translation.stream import DONE, StreamAssembler, sse
from .deps import PrincipalDep, ProviderDep, RecorderDep, SettingsDep

router = APIRouter()
log = get_logger(__name__)


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """The models Prism will serve, in OpenAI's list shape."""
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": spec.id,
                "object": "model",
                "created": now,
                "owned_by": "anthropic",
                "prism": {
                    "tier": spec.tier,
                    "context_window": spec.context_window,
                    "max_output_tokens": spec.max_output_tokens,
                    "supports_effort": spec.supports_effort,
                    "supports_sampling_params": spec.supports_sampling_params,
                },
            }
            for spec in MODELS.values()
        ],
    }


@router.post("/chat/completions")
async def create_chat_completion(
    body: ChatCompletionRequest,
    principal: PrincipalDep,
    provider: ProviderDep,
    recorder: RecorderDep,
    settings: SettingsDep,
) -> Response:
    started = time.perf_counter()

    draft = TraceDraft(
        endpoint="/v1/chat/completions",
        model_requested=body.model,
        model_resolved=body.model,
        stream=body.stream,
        tenant_id=principal.tenant_id,
        api_key_id=principal.api_key_id,
        prompt_version=body.prism.prompt_version if body.prism else None,
        request_body=body.model_dump(mode="json", exclude_none=True),
    )

    def fail(error: PrismError) -> PrismError:
        draft.status = "error"
        draft.error_type = str(error.kind)
        draft.error_message = error.message
        draft.http_status = error.status_code
        draft.latency_ms = int((time.perf_counter() - started) * 1000)
        draft.upstream_response = (
            error.upstream_body if isinstance(error.upstream_body, dict) else None
        )
        recorder.record_background(draft)
        return error

    try:
        spec = resolve_model(body.model)
    except UnknownModelError as exc:
        raise fail(
            PrismError(
                ErrorKind.INVALID_REQUEST, str(exc), status_code=404, param="model"
            )
        ) from exc

    draft.model_resolved = spec.id

    try:
        upstream_body, warnings = request_translator.translate(
            body, spec, default_max_tokens=settings.default_max_tokens
        )
    except PrismError as exc:
        raise fail(exc) from exc

    draft.upstream_request = upstream_body
    if warnings:
        draft.extra["translation_warnings"] = sorted(set(warnings))

    if body.stream:
        frames = _stream_completion(
            body=body,
            spec=spec,
            upstream_body=upstream_body,
            draft=draft,
            provider=provider,
            recorder=recorder,
            settings=settings,
            started=started,
        )
        # Prime the generator before handing it to Starlette. Once the first byte
        # is written the status line is committed and every later failure can
        # only be an SSE error frame — so the upstream connection is opened here,
        # while a 429 or a 529 can still be relayed as a real status code.
        try:
            first_frame = await frames.__anext__()
        except StopAsyncIteration:
            first_frame = None
        except PrismError as exc:
            raise fail(exc) from exc

        return StreamingResponse(
            _prepend(first_frame, frames),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "connection": "keep-alive",
                # Nginx and friends buffer proxied responses by default, which
                # turns a token stream back into one blob at the end.
                "x-accel-buffering": "no",
                "x-prism-model": spec.id,
            },
        )

    try:
        upstream = await provider.create_message(upstream_body)
    except PrismError as exc:
        raise fail(exc) from exc

    completion = response_translator.translate(
        upstream.payload, model_requested=body.model
    )
    usage = response_translator.extract_usage(upstream.payload)
    cost = compute_cost(usage, spec)

    draft.usage = usage
    draft.cost = cost
    draft.http_status = 200
    draft.stop_reason = upstream.payload.get("stop_reason")
    draft.finish_reason = completion.choices[0].finish_reason
    draft.provider_request_id = upstream.provider_request_id
    draft.upstream_latency_ms = upstream.latency_ms
    draft.latency_ms = int((time.perf_counter() - started) * 1000)
    draft.upstream_response = upstream.payload
    draft.response_body = completion.model_dump(mode="json", exclude_none=True)
    draft.extra["rate_limits"] = upstream.rate_limits.as_json()

    recorder.record_background(draft)

    payload = completion.model_dump(mode="json", exclude_none=True)
    return JSONResponse(
        payload,
        headers={
            # The client keeps the model string it sent; these headers say what
            # actually happened, which is what makes a drop-in swap debuggable.
            "x-prism-model": spec.id,
            "x-prism-cost-usd": f"{cost.total_usd:.8f}",
            "x-prism-cache-read-tokens": str(usage.cache_read_input_tokens),
            "x-prism-cache-write-tokens": str(usage.cache_creation_input_tokens),
            "x-prism-upstream-latency-ms": str(upstream.latency_ms),
            # Not clamped at zero: a negative value means the two clocks
            # disagree, and hiding that behind a 0 would make a broken
            # measurement look like a fast one.
            "x-prism-gateway-overhead-ms": str(
                (draft.latency_ms or 0) - upstream.latency_ms
            ),
        },
    )


async def _prepend(
    first: bytes | None, rest: AsyncIterator[bytes]
) -> AsyncIterator[bytes]:
    if first is not None:
        yield first
    async for frame in rest:
        yield frame


def _carries_content(chunk: dict[str, Any]) -> bool:
    delta = chunk["choices"][0]["delta"] if chunk.get("choices") else {}
    return bool(
        delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
    )


async def _stream_completion(
    *,
    body: ChatCompletionRequest,
    spec: ModelSpec,
    upstream_body: dict[str, Any],
    draft: TraceDraft,
    provider: Any,
    recorder: TraceRecorder,
    settings: Any,
    started: float,
) -> AsyncIterator[bytes]:
    """Tee the upstream stream: relay to the client, buffer to reconstruct.

    You cannot cache, score, or cost a response you have not finished reading, so
    the buffer runs alongside the relay rather than after it. What the buffer
    produces is a complete Anthropic ``Message``, which then goes through exactly
    the same translation and cost path as a non-streamed response.
    """
    assembler = StreamAssembler(
        model_requested=body.model,
        completion_id=f"chatcmpl-{uuid.uuid4().hex}",
        thinking_policy=settings.trace_thinking,
    )
    include_usage = bool(body.stream_options and body.stream_options.include_usage)

    # A large thinking budget legitimately delays the first *visible* token, so a
    # fixed first-token budget would kill healthy requests. The budget is
    # conditioned on what was actually sent upstream.
    first_token_budget = (
        settings.thinking_first_token_timeout_s
        if "thinking" in upstream_body
        else settings.first_token_timeout_s
    )

    outcome = "ok"
    stream_error: PrismError | None = None

    try:
        async with provider.stream_message(
            upstream_body,
            first_token_timeout_s=first_token_budget,
            total_timeout_s=settings.request_timeout_s,
        ) as handle:
            draft.provider_request_id = handle.provider_request_id
            draft.extra["rate_limits"] = handle.rate_limits.as_json()

            async for event in handle.events():
                for chunk in assembler.handle(event):
                    if draft.ttft_ms is None and _carries_content(chunk):
                        draft.ttft_ms = int((time.perf_counter() - started) * 1000)
                    yield sse(chunk)

            if include_usage:
                yield sse(assembler.usage_chunk())
            yield DONE

    except (asyncio.CancelledError, GeneratorExit):
        # The client hung up mid-stream. Leaving the context manager closes the
        # upstream connection, which stops generation and stops the meter — see
        # `stream_drain_on_disconnect` for the other side of that trade.
        outcome = "client_disconnect"
        raise
    except PrismError as exc:
        stream_error = exc
        outcome = "error"
        if assembler.message_id:
            # Bytes have already reached the client, so the status line is spent
            # and an SSE error frame is the only channel left. A client that only
            # reads `choices` will see a truncated answer; one that checks for
            # `error` sees why.
            yield sse({"error": exc.to_response().model_dump()["error"]})
            yield DONE
        else:
            raise
    finally:
        _finalize_stream_trace(
            assembler=assembler,
            draft=draft,
            spec=spec,
            body=body,
            recorder=recorder,
            started=started,
            outcome=outcome,
            error=stream_error,
        )


def _finalize_stream_trace(
    *,
    assembler: StreamAssembler,
    draft: TraceDraft,
    spec: ModelSpec,
    body: ChatCompletionRequest,
    recorder: TraceRecorder,
    started: float,
    outcome: str,
    error: PrismError | None,
) -> None:
    message = assembler.to_message()
    usage = assembler.token_usage()

    draft.usage = usage
    draft.cost = compute_cost(usage, spec)
    draft.latency_ms = int((time.perf_counter() - started) * 1000)
    # upstream_latency_ms stays null for streams. For a streamed response the
    # gateway's own cost is spread across the whole relay rather than
    # bracketing one upstream call, so reporting a number here would be a
    # claim the measurement does not support. ttft_ms is the honest metric.
    draft.stop_reason = assembler.stop_reason
    draft.finish_reason = to_finish_reason(assembler.stop_reason)
    draft.upstream_response = message

    if outcome == "ok":
        draft.status = "ok"
        draft.http_status = 200
        completion = response_translator.translate(
            message, model_requested=body.model, completion_id=assembler.completion_id
        )
        draft.response_body = completion.model_dump(mode="json", exclude_none=True)
    else:
        draft.status = outcome
        draft.http_status = error.status_code if error else 499
        draft.error_type = str(error.kind) if error else "client_disconnect"
        draft.error_message = (
            error.message if error else "Client disconnected before the stream finished."
        )

    # A partial response is not a cheap cache entry, it is a wrong one. This flag
    # is what the week-8 cache writer reads before storing anything.
    draft.extra |= {
        "stream_complete": assembler.complete,
        "cacheable": assembler.complete,
        "truncated_tool_input": assembler.truncated_tool_input,
        "blocks": len(message["content"]),
    }
    recorder.record_background(draft)
