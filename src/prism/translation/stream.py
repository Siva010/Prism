"""Streaming reassembly: Anthropic typed events -> OpenAI flat deltas.

This is the part of the protocol translation that is genuinely harder than a
field rename, and it is worth being precise about why.

OpenAI's stream is a flat sequence of `choices[0].delta` fragments — you can
concatenate `delta.content` into the final answer with a string accumulator.
Anthropic's stream is *block-structured*: `message_start`, then for each content
block a `content_block_start` / `content_block_delta`* / `content_block_stop`
triple, then `message_delta` and `message_stop`. Blocks carry types (text,
thinking, tool_use) and a block index, and tool inputs arrive as a stream of
partial JSON fragments that are not valid JSON until the block closes.

So the buffer is a **state machine over open content blocks**, not a string. It
does two things at once:

1. emits OpenAI chunks as events arrive, so the client sees tokens immediately;
2. reconstructs the complete Anthropic ``Message`` object — byte-identical in
   shape to what the non-streaming endpoint returns.

(2) is what makes the rest of the system work: the reconstructed message goes
through exactly the same ``response.translate`` / ``extract_usage`` /
``compute_cost`` path as a non-streamed one, so traces, cost accounting, and
(from week 8) cache values do not need a second implementation that can drift.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from ..cost import TokenUsage
from .mapping import to_finish_reason
from .response import to_openai_usage

ThinkingPolicy = Literal["persist", "redact", "drop"]


@dataclass
class _Block:
    """One open (or closed) content block."""

    index: int
    type: str
    # text / thinking blocks accumulate strings; tool_use accumulates partial JSON.
    text: str = ""
    thinking: str = ""
    signature: str = ""
    partial_json: str = ""
    tool_id: str = ""
    tool_name: str = ""
    tool_call_index: int | None = None
    closed: bool = False

    def to_content_block(self) -> dict[str, Any] | None:
        if self.type == "text":
            return {"type": "text", "text": self.text}
        if self.type == "thinking":
            block: dict[str, Any] = {"type": "thinking", "thinking": self.thinking}
            if self.signature:
                block["signature"] = self.signature
            return block
        if self.type == "tool_use":
            return {
                "type": "tool_use",
                "id": self.tool_id,
                "name": self.tool_name,
                "input": self._parsed_input(),
            }
        return None

    def _parsed_input(self) -> dict[str, Any]:
        if not self.partial_json.strip():
            return {}
        try:
            parsed = json.loads(self.partial_json)
        except json.JSONDecodeError:
            # Truncated mid-block — the stream was cut before the tool input was
            # complete. Preserved verbatim rather than guessed at, and the
            # assembler reports `truncated` so nothing downstream treats this as
            # a usable tool call.
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @property
    def tool_input_is_valid(self) -> bool:
        if self.type != "tool_use":
            return True
        if not self.partial_json.strip():
            return True
        try:
            json.loads(self.partial_json)
        except json.JSONDecodeError:
            return False
        return True


@dataclass
class StreamAssembler:
    """Tee: emit OpenAI chunks downstream while rebuilding the Anthropic message."""

    model_requested: str
    completion_id: str
    created: int = field(default_factory=lambda: int(time.time()))
    thinking_policy: ThinkingPolicy = "persist"

    message_id: str = ""
    model_resolved: str = ""
    stop_reason: str | None = None
    stop_sequence: str | None = None
    saw_message_stop: bool = False

    _blocks: dict[int, _Block] = field(default_factory=dict)
    _usage: dict[str, int] = field(default_factory=dict)
    _next_tool_call_index: int = 0
    _role_chunk_sent: bool = False
    _first_content_at: float | None = None

    # --- ingest ----------------------------------------------------------

    def handle(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Consume one upstream event; return the OpenAI chunks it produces."""
        kind = event.get("type")
        handler = getattr(self, f"_on_{kind}", None)
        if handler is None:
            # ping, and any event type added upstream after this was written.
            # Unknown events are ignored rather than fatal: a gateway that dies
            # on a new event type breaks every client at once.
            return []
        chunks: list[dict[str, Any]] = handler(event)
        return chunks

    def _on_message_start(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        message = event.get("message") or {}
        self.message_id = message.get("id", "")
        self.model_resolved = message.get("model", "")
        usage = message.get("usage") or {}
        # The input-side counts arrive here and never change; output_tokens on
        # message_start is a running total that message_delta supersedes.
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
        ):
            if usage.get(key) is not None:
                self._usage[key] = int(usage[key])
        return [self._chunk({"role": "assistant", "content": ""})]

    def _on_content_block_start(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        index = int(event.get("index", 0))
        raw = event.get("content_block") or {}
        block = _Block(index=index, type=raw.get("type", "text"))

        if block.type == "text":
            block.text = raw.get("text", "") or ""
        elif block.type == "thinking":
            block.thinking = raw.get("thinking", "") or ""
        elif block.type == "tool_use":
            block.tool_id = raw.get("id", "")
            block.tool_name = raw.get("name", "")
            block.tool_call_index = self._next_tool_call_index
            self._next_tool_call_index += 1

        self._blocks[index] = block

        if block.type == "tool_use":
            # OpenAI clients key tool calls by their position in `tool_calls`,
            # which is not the Anthropic block index — a text block before the
            # first tool_use would shift them by one.
            return [
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": block.tool_call_index,
                                "id": block.tool_id,
                                "type": "function",
                                "function": {"name": block.tool_name, "arguments": ""},
                            }
                        ]
                    }
                )
            ]
        return []

    def _on_content_block_delta(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        index = int(event.get("index", 0))
        block = self._blocks.get(index)
        if block is None:
            return []
        delta = event.get("delta") or {}
        delta_type = delta.get("type")

        if delta_type == "text_delta":
            text = delta.get("text", "")
            block.text += text
            self._mark_first_content()
            return [self._chunk({"content": text})] if text else []

        if delta_type == "thinking_delta":
            thinking = delta.get("thinking", "")
            block.thinking += thinking
            self._mark_first_content()
            # Non-standard field, but the convention OpenAI-compatible servers
            # settled on for reasoning models. Empty unless the request asked for
            # display="summarized".
            return [self._chunk({"reasoning_content": thinking})] if thinking else []

        if delta_type == "signature_delta":
            # Opaque verification blob for multi-turn thinking. Never forwarded
            # to the client and never persisted — it is large and useless outside
            # a follow-up request.
            block.signature += delta.get("signature", "")
            return []

        if delta_type == "input_json_delta":
            fragment = delta.get("partial_json", "")
            block.partial_json += fragment
            self._mark_first_content()
            if not fragment or block.tool_call_index is None:
                return []
            return [
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": block.tool_call_index,
                                "function": {"arguments": fragment},
                            }
                        ]
                    }
                )
            ]

        return []

    def _on_content_block_stop(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        block = self._blocks.get(int(event.get("index", 0)))
        if block is not None:
            block.closed = True
        return []

    def _on_message_delta(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        delta = event.get("delta") or {}
        self.stop_reason = delta.get("stop_reason", self.stop_reason)
        self.stop_sequence = delta.get("stop_sequence", self.stop_sequence)
        usage = event.get("usage") or {}
        if usage.get("output_tokens") is not None:
            self._usage["output_tokens"] = int(usage["output_tokens"])
        # Some responses report additional cache counts only here.
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if usage.get(key) is not None:
                self._usage[key] = int(usage[key])
        return [self._chunk({}, finish_reason=to_finish_reason(self.stop_reason))]

    def _on_message_stop(self, _: dict[str, Any]) -> list[dict[str, Any]]:
        self.saw_message_stop = True
        return []

    def _mark_first_content(self) -> None:
        if self._first_content_at is None:
            self._first_content_at = time.perf_counter()

    def _chunk(self, delta: dict[str, Any], *, finish_reason: str | None = None) -> dict[str, Any]:
        return {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_requested,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    # --- output ----------------------------------------------------------

    @property
    def complete(self) -> bool:
        """True only for a stream that ran to `message_stop` with usable blocks."""
        return self.saw_message_stop and all(
            b.closed and b.tool_input_is_valid for b in self._blocks.values()
        )

    @property
    def truncated_tool_input(self) -> bool:
        return any(not b.tool_input_is_valid for b in self._blocks.values())

    def usage_chunk(self) -> dict[str, Any]:
        """Terminal chunk for clients that sent `stream_options.include_usage`."""
        usage = to_openai_usage(self.token_usage())
        return {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_requested,
            "choices": [],
            "usage": usage.model_dump(exclude_none=True),
        }

    def token_usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self._usage.get("input_tokens", 0),
            cache_read_input_tokens=self._usage.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=self._usage.get("cache_creation_input_tokens", 0),
            output_tokens=self._usage.get("output_tokens", 0),
        )

    def to_message(self) -> dict[str, Any]:
        """The reconstructed Anthropic Message.

        Same shape as the non-streaming response, so one translation path serves
        both. Blocks are emitted in index order — a dict keyed by index, not an
        append-only list, because nothing guarantees blocks arrive in order.
        """
        content: list[dict[str, Any]] = []
        for index in sorted(self._blocks):
            block = self._blocks[index]
            rendered = block.to_content_block()
            if rendered is None:
                continue
            if rendered["type"] == "thinking":
                if self.thinking_policy == "drop":
                    continue
                if self.thinking_policy == "redact":
                    rendered = {
                        "type": "thinking",
                        "thinking": f"[redacted: {len(block.thinking)} chars]",
                    }
                else:
                    rendered.pop("signature", None)
            content.append(rendered)

        return {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model_resolved,
            "content": content,
            "stop_reason": self.stop_reason,
            "stop_sequence": self.stop_sequence,
            "usage": dict(self._usage),
        }


def sse(payload: dict[str, Any] | str) -> bytes:
    """Frame one value as an SSE `data:` event."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n".encode()
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


DONE = sse("[DONE]")
