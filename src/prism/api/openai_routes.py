"""OpenAI-compatible ingress.

Week 1-2 scope: non-streaming `/v1/chat/completions`. Streaming is rejected with
an explicit 501 rather than silently downgraded to a buffered response — a client
that asked for tokens as they arrive and got one blob at the end has been given a
different product, and hiding that would make the week-3 stream work look done.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..cost import compute_cost
from ..logging_setup import get_logger
from ..registry import MODELS, UnknownModelError, resolve_model
from ..schemas.errors import ErrorKind, PrismError
from ..schemas.openai import ChatCompletionRequest
from ..tracing.recorder import TraceDraft
from ..translation import request as request_translator
from ..translation import response as response_translator
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
) -> JSONResponse:
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

    if body.stream:
        raise fail(
            PrismError(
                ErrorKind.INVALID_REQUEST,
                "Streaming is not implemented yet (week 3). Retry with "
                "`stream: false`.",
                status_code=501,
                param="stream",
            )
        )

    try:
        upstream_body, warnings = request_translator.translate(
            body, spec, default_max_tokens=settings.default_max_tokens
        )
    except PrismError as exc:
        raise fail(exc) from exc

    draft.upstream_request = upstream_body
    if warnings:
        draft.extra["translation_warnings"] = sorted(set(warnings))

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
