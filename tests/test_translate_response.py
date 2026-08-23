"""Anthropic -> OpenAI response translation."""

from __future__ import annotations

import json

from prism.translation.response import extract_usage, translate


def message(**overrides):
    payload = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": "Paris."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    payload.update(overrides)
    return payload


def test_text_blocks_collapse_into_a_single_content_string():
    out = translate(
        message(
            content=[
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world."},
            ]
        ),
        model_requested="gpt-4o",
    )
    assert out.choices[0].message.content == "Hello world."


def test_the_client_gets_back_the_model_string_it_sent():
    # A client that pinned "gpt-4o" should not find its own string rewritten; the
    # model actually served travels in x-prism-model and on the trace.
    out = translate(message(), model_requested="gpt-4o")
    assert out.model == "gpt-4o"


def test_tool_use_blocks_become_tool_calls_with_stringified_arguments():
    out = translate(
        message(
            stop_reason="tool_use",
            content=[
                {"type": "text", "text": "Looking it up."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "Paris", "unit": "c"},
                },
            ],
        ),
        model_requested="claude-opus-5",
    )
    call = out.choices[0].message.tool_calls[0]
    assert call.id == "toolu_01"  # preserved: it is the join key on the next turn
    assert json.loads(call.function.arguments) == {"city": "Paris", "unit": "c"}
    assert out.choices[0].finish_reason == "tool_calls"


def test_content_is_null_when_the_turn_is_only_a_tool_call():
    out = translate(
        message(
            stop_reason="tool_use",
            content=[{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
        ),
        model_requested="claude-opus-5",
    )
    assert out.choices[0].message.content is None


def test_stop_reasons_map_to_finish_reasons():
    cases = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
    }
    for stop_reason, expected in cases.items():
        out = translate(message(stop_reason=stop_reason), model_requested="m")
        assert out.choices[0].finish_reason == expected, stop_reason


def test_summarized_thinking_lands_in_reasoning_content():
    out = translate(
        message(
            content=[
                {"type": "thinking", "thinking": "Considering options."},
                {"type": "text", "text": "Paris."},
            ]
        ),
        model_requested="m",
    )
    assert out.choices[0].message.reasoning_content == "Considering options."
    assert out.choices[0].message.content == "Paris."


def test_empty_thinking_blocks_are_not_surfaced():
    # display defaults to "omitted", which streams thinking blocks with empty text.
    out = translate(
        message(content=[{"type": "thinking", "thinking": ""}, {"type": "text", "text": "x"}]),
        model_requested="m",
    )
    assert out.choices[0].message.reasoning_content is None


def test_prompt_tokens_include_cached_and_written_input():
    usage = extract_usage(
        message(
            usage={
                "input_tokens": 100,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 50,
                "output_tokens": 20,
            }
        )
    )
    assert usage.billable_input_tokens == 1050
    assert usage.prefix_cache_hit_rate == 900 / 1050

    out = translate(
        message(
            usage={
                "input_tokens": 100,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 50,
                "output_tokens": 20,
            }
        ),
        model_requested="m",
    )
    assert out.usage.prompt_tokens == 1050
    assert out.usage.prompt_tokens_details.cached_tokens == 900
    # OpenAI's usage object has no slot for cache writes; dropping them would
    # make the response uncostable.
    assert out.usage.prism_cache_creation_input_tokens == 50


def test_write_read_ratio_flags_a_badly_placed_breakpoint():
    usage = extract_usage(
        message(usage={"input_tokens": 0, "cache_creation_input_tokens": 800, "output_tokens": 5})
    )
    # Writes with no reads: the breakpoint sits on volatile content and every
    # request pays the write premium for nothing.
    assert usage.write_read_ratio == float("inf")
