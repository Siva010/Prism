"""Anthropic Messages response -> OpenAI Chat Completions response.

The asymmetry worth naming: Anthropic returns a *list of typed content blocks*
and OpenAI returns a message with a flat string plus a separate `tool_calls`
array. Collapsing the former into the latter is lossy, so anything that does not
fit the OpenAI shape (thinking summaries, cache-write token counts, the precise
stop reason) is either carried in a namespaced extra field or recorded on the
trace rather than dropped.

Tool-use ids are passed through unmodified. They are the join key on the next
turn — the client echoes them back as `tool_call_id`, and the request translator
turns them straight back into `tool_use_id` — so rewriting them would break
multi-turn tool use.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ..cost import TokenUsage
from ..schemas.openai import (
    ChatCompletionResponse,
    Choice,
    CompletionTokensDetails,
    FunctionCall,
    PromptTokensDetails,
    ResponseMessage,
    ToolCall,
    Usage,
)
from .mapping import to_finish_reason


def extract_usage(payload: dict[str, Any]) -> TokenUsage:
    """Normalize the Anthropic `usage` object into the five token classes."""
    usage = payload.get("usage") or {}
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        # Thinking tokens are billed inside output_tokens and are not itemized in
        # the usage object; the field exists so a future estimator can fill it.
        thinking_tokens=0,
    )


def to_openai_usage(usage: TokenUsage) -> Usage:
    prompt_tokens = usage.billable_input_tokens
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=prompt_tokens + usage.output_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=usage.cache_read_input_tokens),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=usage.thinking_tokens),
        # OpenAI's usage object has no slot for cache *writes*. Omitting them
        # would make the returned usage uncostable, so they travel in a
        # namespaced field and are always authoritative on the trace.
        prism_cache_creation_input_tokens=usage.cache_creation_input_tokens,
    )


def translate(
    payload: dict[str, Any],
    *,
    model_requested: str,
    completion_id: str | None = None,
    created: int | None = None,
) -> ChatCompletionResponse:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in payload.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            text_parts.append(block.get("text", ""))
        elif kind == "thinking":
            # Empty unless the request asked for display="summarized"; the raw
            # chain of thought is never returned.
            if block.get("thinking"):
                thinking_parts.append(block["thinking"])
        elif kind == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block["id"],
                    function=FunctionCall(
                        name=block["name"],
                        # OpenAI clients parse `arguments` as a JSON string. Sorted
                        # keys keep the serialization stable, which matters once
                        # this response becomes a cache value.
                        arguments=json.dumps(
                            block.get("input") or {}, sort_keys=True, separators=(",", ":")
                        ),
                    ),
                )
            )

    content = "".join(text_parts)
    message = ResponseMessage(
        content=content if content else None,
        tool_calls=tool_calls or None,
        reasoning_content="\n".join(thinking_parts) or None,
    )

    return ChatCompletionResponse(
        id=completion_id or f"chatcmpl-{uuid.uuid4().hex}",
        created=created or int(time.time()),
        # Echo what the client asked for, not what was served. The resolved model
        # is on the trace and in the `x-prism-model` response header — a client
        # that pinned "gpt-4o" should not have its own string changed underneath it.
        model=model_requested,
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason=to_finish_reason(payload.get("stop_reason")),
            )
        ],
        usage=to_openai_usage(extract_usage(payload)),
    )
