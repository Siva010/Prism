"""OpenAI Chat Completions request -> Anthropic Messages request.

This is not a field rename. Four structural differences do real work:

* the system prompt moves from a message into a top-level field;
* tool calls and tool results move from message *fields* into content *blocks*;
* Anthropic requires strictly alternating user/assistant turns, so several
  OpenAI messages can collapse into one;
* ``max_tokens`` is optional upstream of us and required downstream.

Model capability differences are handled here too — sampling parameters are
accepted from Opus 5 / Sonnet 5 clients and dropped rather than forwarded,
because forwarding them returns a 400 on those models and works on Haiku. A
gateway that passes them through has a request that succeeds or fails depending
on which tier the router happened to pick.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any

from ..registry import ModelSpec
from ..schemas.errors import ErrorKind, PrismError
from ..schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPart,
    NamedToolChoice,
    ToolCall,
)
from .mapping import TOOL_CHOICE_MAP

_DATA_URI = re.compile(r"^data:(?P<media_type>[\w./+-]+);base64,(?P<data>.+)$", re.S)

# Parameters Prism accepts from a client but cannot express upstream. Silently
# ignoring them is the wrong default: a client that sets `n=3` and receives one
# choice has been given a wrong answer, not a degraded one.
_UNSUPPORTED: dict[str, str] = {
    "n": "Prism returns exactly one choice per request; `n` > 1 is not supported.",
    "logprobs": "Anthropic's Messages API does not expose logprobs.",
    "frequency_penalty": "No Anthropic equivalent.",
    "presence_penalty": "No Anthropic equivalent.",
    "logit_bias": "No Anthropic equivalent.",
}


class TranslationWarning(str):
    """A lossy-but-safe translation decision, recorded on the trace."""


def _invalid(message: str, param: str | None = None) -> PrismError:
    return PrismError(ErrorKind.INVALID_REQUEST, message, status_code=400, param=param)


def _content_to_blocks(content: str | list[ContentPart] | None) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        # Anthropic rejects empty text blocks; an empty string is best expressed
        # as no block at all.
        return [{"type": "text", "text": content}] if content else []

    blocks: list[dict[str, Any]] = []
    for part in content:
        if part.type == "text":
            if part.text:
                blocks.append({"type": "text", "text": part.text})
            continue
        url = part.image_url.url
        match = _DATA_URI.match(url)
        if match:
            data = match.group("data")
            try:
                base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise _invalid(
                    f"image_url data URI is not valid base64: {exc}", "messages"
                ) from exc
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": match.group("media_type"),
                        "data": data,
                    },
                }
            )
        elif url.startswith(("http://", "https://")):
            blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            raise _invalid(
                "image_url must be an http(s) URL or a base64 data URI.", "messages"
            )
    return blocks


def _tool_result_content(content: str | list[ContentPart] | None) -> str | list[Any]:
    blocks = _content_to_blocks(content)
    if not blocks:
        return ""
    if all(b["type"] == "text" for b in blocks):
        return "\n".join(b["text"] for b in blocks)
    return blocks


def _extract_system(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], list[ChatMessage]]:
    """Hoist every system/developer message into the top-level `system` field.

    OpenAI permits system messages anywhere in the array; Anthropic has exactly
    one system slot, rendered before all messages. Hoisting preserves relative
    order between system messages but not their position relative to the
    conversation — recorded as a translation warning when a system message was
    not at the head of the array.
    """
    system_blocks: list[dict[str, Any]] = []
    rest: list[ChatMessage] = []
    for message in messages:
        if message.role in ("system", "developer"):
            blocks = _content_to_blocks(message.content)
            if not all(b["type"] == "text" for b in blocks):
                raise _invalid("System messages must be text only.", "messages")
            system_blocks.extend(blocks)
        else:
            rest.append(message)
    return system_blocks, rest


def _parse_arguments(call: ToolCall) -> dict[str, Any]:
    raw = call.function.arguments
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _invalid(
            f"tool_call {call.id!r} has arguments that are not valid JSON: {exc}",
            "messages",
        ) from exc
    if not isinstance(parsed, dict):
        raise _invalid(
            f"tool_call {call.id!r} arguments must decode to a JSON object.", "messages"
        )
    return parsed


def _build_messages(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], list[TranslationWarning]]:
    """Fold OpenAI messages into strictly alternating Anthropic turns."""
    warnings: list[TranslationWarning] = []
    out: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for message in messages:
        if message.role == "tool":
            if not message.tool_call_id:
                raise _invalid(
                    "A message with role 'tool' requires 'tool_call_id'.", "messages"
                )
            # Tool results are user-turn content blocks upstream. Consecutive tool
            # messages therefore merge into a single user turn — emitting one user
            # message each would break alternation and would train the model out of
            # making parallel tool calls.
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": _tool_result_content(message.content),
                    }
                ],
            )
            continue

        if message.role == "user":
            blocks = _content_to_blocks(message.content)
            if not blocks:
                raise _invalid("User messages must have content.", "messages")
            append("user", blocks)
            continue

        if message.role == "assistant":
            blocks = _content_to_blocks(message.content)
            for call in message.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.function.name,
                        "input": _parse_arguments(call),
                    }
                )
            if not blocks:
                # An assistant turn with neither content nor tool calls carries no
                # information upstream and would break alternation.
                warnings.append(TranslationWarning("dropped_empty_assistant_message"))
                continue
            append("assistant", blocks)
            continue

        raise _invalid(f"Unsupported message role {message.role!r}.", "messages")

    if not out:
        raise _invalid(
            "`messages` must contain at least one non-system message.", "messages"
        )
    if out[0]["role"] != "user":
        raise _invalid("The first non-system message must have role 'user'.", "messages")
    return out, warnings


def _build_tools(req: ChatCompletionRequest) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in req.tools or []:
        fn = tool.function
        definition: dict[str, Any] = {
            "name": fn.name,
            "input_schema": fn.parameters or {"type": "object", "properties": {}},
        }
        if fn.description:
            definition["description"] = fn.description
        if fn.strict:
            # Grammar-constrained tool inputs. The provider enforces only a subset
            # of JSON Schema; the post-validation half lands in week 8.
            definition["strict"] = True
        tools.append(definition)
    return tools


def _build_tool_choice(req: ChatCompletionRequest) -> dict[str, Any] | None:
    choice = req.tool_choice
    if choice is None:
        return None
    if isinstance(choice, NamedToolChoice):
        return {"type": "tool", "name": choice.function.name}
    if choice == "none":
        # Handled by the caller by omitting `tools` entirely — see translate().
        return None
    mapped = TOOL_CHOICE_MAP.get(choice)
    if mapped is None:
        raise _invalid(f"Unsupported tool_choice {choice!r}.", "tool_choice")
    return mapped


def _build_output_config(
    req: ChatCompletionRequest, spec: ModelSpec, warnings: list[TranslationWarning]
) -> dict[str, Any]:
    config: dict[str, Any] = {}

    fmt = req.response_format
    if fmt is not None and fmt.type == "json_schema":
        if fmt.json_schema is None or fmt.json_schema.schema_ is None:
            raise _invalid(
                "response_format.json_schema requires a `schema`.", "response_format"
            )
        config["format"] = {"type": "json_schema", "schema": fmt.json_schema.schema_}
    elif fmt is not None and fmt.type == "json_object":
        # `json_object` has no schema to compile a grammar from. Week 8 replaces
        # this with a validate-and-repair path; for now it is a documented no-op
        # rather than a silent one.
        warnings.append(TranslationWarning("response_format_json_object_unenforced"))

    effort = req.prism.effort if req.prism else None
    if effort:
        if spec.supports_effort:
            config["effort"] = effort
        else:
            warnings.append(TranslationWarning("effort_dropped_unsupported_model"))

    return config


def _build_thinking(
    req: ChatCompletionRequest, spec: ModelSpec, warnings: list[TranslationWarning]
) -> dict[str, Any] | None:
    want_thinking = req.prism.thinking if req.prism else None
    if want_thinking is None:
        return None
    if spec.thinking == "adaptive":
        if not want_thinking:
            # Thinking is on by default on these models and disabling it has its
            # own failure modes; lowering effort is the better dial.
            warnings.append(TranslationWarning("thinking_disable_ignored"))
            return None
        return {"type": "adaptive", "display": "summarized"}
    if spec.thinking == "budget_tokens":
        # Pre-4.6 tiers still take an explicit budget. Axis 2 of the router
        # (week 9) is what sets that number, so until then this is a no-op.
        warnings.append(TranslationWarning("thinking_budget_not_set"))
        return None
    warnings.append(TranslationWarning("thinking_unsupported_model"))
    return None


def translate(
    req: ChatCompletionRequest,
    spec: ModelSpec,
    *,
    default_max_tokens: int,
) -> tuple[dict[str, Any], list[TranslationWarning]]:
    """Return an Anthropic Messages request body and any lossy-translation notes."""
    raw = req.model_dump(exclude_unset=True)
    for field, reason in _UNSUPPORTED.items():
        value = raw.get(field)
        if value is None:
            continue
        if field == "n" and value == 1:
            continue
        raise _invalid(reason, field)

    warnings: list[TranslationWarning] = []

    if req.messages and req.messages[-1].role == "assistant":
        # An assistant message in the final position is a response prefill.
        # Current Claude models reject it with a 400, so failing here produces a
        # better error than relaying the upstream one.
        raise PrismError(
            ErrorKind.TRANSLATION,
            "A trailing assistant message (response prefill) is not supported by "
            "current Claude models. Use response_format or a system instruction to "
            "constrain the output shape instead.",
            status_code=400,
            param="messages",
        )

    system_blocks, conversation = _extract_system(req.messages)
    if system_blocks and req.messages[0].role not in ("system", "developer"):
        warnings.append(
            TranslationWarning("system_message_hoisted_from_mid_conversation")
        )

    messages, fold_warnings = _build_messages(conversation)
    warnings.extend(fold_warnings)

    max_tokens = req.effective_max_tokens or default_max_tokens
    if max_tokens > spec.max_output_tokens:
        warnings.append(TranslationWarning("max_tokens_clamped_to_model_limit"))
        max_tokens = spec.max_output_tokens
    if max_tokens < 1:
        raise _invalid("max_tokens must be a positive integer.", "max_tokens")

    body: dict[str, Any] = {
        "model": spec.id,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_blocks:
        body["system"] = system_blocks

    if req.stop is not None:
        body["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop

    # Sampling parameters exist on Haiku 4.5 and are rejected with a 400 on
    # Opus 5 / Sonnet 5. Dropping them keeps one client request valid across the
    # whole routing ladder.
    for name, value in (("temperature", req.temperature), ("top_p", req.top_p)):
        if value is None:
            continue
        if spec.supports_sampling_params:
            body[name] = value
        else:
            warnings.append(TranslationWarning(f"{name}_dropped_unsupported_model"))

    if req.tools and req.tool_choice != "none":
        body["tools"] = _build_tools(req)
        tool_choice = _build_tool_choice(req)
        if tool_choice is None and req.parallel_tool_calls is False:
            tool_choice = {"type": "auto"}
        if tool_choice is not None:
            if req.parallel_tool_calls is False:
                tool_choice = {**tool_choice, "disable_parallel_tool_use": True}
            body["tool_choice"] = tool_choice
    elif req.tools and req.tool_choice == "none":
        # There is no "offer these tools but do not call them" mode to rely on
        # here, so the tool list is omitted. Recorded, because it changes what the
        # model can see.
        warnings.append(TranslationWarning("tools_omitted_for_tool_choice_none"))

    output_config = _build_output_config(req, spec, warnings)
    if output_config:
        body["output_config"] = output_config

    thinking = _build_thinking(req, spec, warnings)
    if thinking:
        body["thinking"] = thinking

    if req.user:
        body["metadata"] = {"user_id": req.user}

    return body, warnings
