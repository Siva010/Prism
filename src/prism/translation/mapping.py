"""Vocabulary mappings between the two protocols."""

from __future__ import annotations

from ..schemas.openai import FinishReason

# Anthropic stop_reason -> OpenAI finish_reason.
#
# The mapping is lossy in one direction that matters: `end_turn`, `stop_sequence`,
# and `pause_turn` all collapse to "stop", so Prism records the Anthropic value
# verbatim on the trace alongside the OpenAI value it returned. A dashboard that
# only had "stop" could not tell a natural completion from a paused server-tool
# turn.
STOP_REASON_TO_FINISH_REASON: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def to_finish_reason(stop_reason: str | None) -> FinishReason | None:
    if stop_reason is None:
        return None
    return STOP_REASON_TO_FINISH_REASON.get(stop_reason, "stop")


# OpenAI tool_choice -> Anthropic tool_choice.
#
# "none" has no direct equivalent that is safe to assume, so the request
# translator drops the tool list instead; see translation/request.py.
TOOL_CHOICE_MAP: dict[str, dict[str, str]] = {
    "auto": {"type": "auto"},
    "required": {"type": "any"},
}
