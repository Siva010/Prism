"""OpenAI -> Anthropic request translation."""

from __future__ import annotations

import base64

import pytest

from prism.registry import MODELS
from prism.schemas.errors import ErrorKind, PrismError
from prism.schemas.openai import ChatCompletionRequest
from prism.translation.request import translate

OPUS = MODELS["claude-opus-5"]
HAIKU = MODELS["claude-haiku-4-5"]


def build(**kwargs) -> ChatCompletionRequest:
    payload = {"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(kwargs)
    return ChatCompletionRequest.model_validate(payload)


def tr(req: ChatCompletionRequest, spec=OPUS):
    return translate(req, spec, default_max_tokens=4096)


def test_system_message_is_hoisted_to_top_level():
    body, warnings = tr(
        build(
            messages=[
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    assert body["system"] == [{"type": "text", "text": "You are terse."}]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert warnings == []


def test_multiple_system_messages_concatenate_in_order():
    body, _ = tr(
        build(
            messages=[
                {"role": "system", "content": "First."},
                {"role": "developer", "content": "Second."},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    assert [b["text"] for b in body["system"]] == ["First.", "Second."]


def test_mid_conversation_system_message_is_hoisted_with_a_warning():
    # Hoisting is correct but lossy — the instruction moves in front of history
    # it was meant to follow — so it has to show up on the trace.
    _, warnings = tr(
        build(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "Be terse now."},
                {"role": "user", "content": "and now?"},
            ]
        )
    )
    assert "system_message_hoisted_from_mid_conversation" in warnings


def test_max_tokens_is_defaulted_because_anthropic_requires_it():
    body, _ = tr(build())
    assert body["max_tokens"] == 4096


def test_max_tokens_is_clamped_to_the_model_ceiling():
    body, warnings = tr(build(max_tokens=999_999))
    assert body["max_tokens"] == OPUS.max_output_tokens
    assert "max_tokens_clamped_to_model_limit" in warnings


def test_max_completion_tokens_is_accepted():
    body, _ = tr(build(max_completion_tokens=256))
    assert body["max_tokens"] == 256


def test_sampling_params_are_dropped_on_models_that_reject_them():
    # Forwarding temperature to Opus 5 is a 400. The same client request must work
    # on every tier the router can pick.
    body, warnings = tr(build(temperature=0.7, top_p=0.9), OPUS)
    assert "temperature" not in body and "top_p" not in body
    assert "temperature_dropped_unsupported_model" in warnings
    assert "top_p_dropped_unsupported_model" in warnings


def test_sampling_params_are_forwarded_where_they_are_supported():
    body, warnings = tr(build(model="claude-haiku-4-5", temperature=0.7), HAIKU)
    assert body["temperature"] == 0.7
    assert warnings == []


def test_assistant_tool_calls_become_tool_use_blocks():
    body, _ = tr(
        build(
            messages=[
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "toolu_01",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_01", "content": "18C"},
                {"role": "user", "content": "thanks"},
            ]
        )
    )
    assistant = body["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "Checking."}
    assert assistant["content"][1] == {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "get_weather",
        # The JSON *string* on the wire becomes a JSON object upstream.
        "input": {"city": "Paris"},
    }


def test_parallel_tool_results_merge_into_one_user_turn():
    # Emitting one user message per tool result would break alternation and
    # teaches the model to stop making parallel calls.
    body, _ = tr(
        build(
            messages=[
                {"role": "user", "content": "weather in Paris and Berlin?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "toolu_01",
                            "function": {"name": "w", "arguments": '{"city":"Paris"}'},
                        },
                        {
                            "id": "toolu_02",
                            "function": {"name": "w", "arguments": '{"city":"Berlin"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_01", "content": "18C"},
                {"role": "tool", "tool_call_id": "toolu_02", "content": "12C"},
                {"role": "user", "content": "thanks"},
            ]
        )
    )
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user"]
    results = body["messages"][2]["content"]
    assert [b["tool_use_id"] for b in results[:2]] == ["toolu_01", "toolu_02"]
    assert results[2] == {"type": "text", "text": "thanks"}


def test_empty_tool_arguments_become_an_empty_object():
    body, _ = tr(
        build(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "toolu_01", "function": {"name": "ping", "arguments": ""}}
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_01", "content": "pong"},
            ]
        )
    )
    assert body["messages"][1]["content"][0]["input"] == {}


def test_malformed_tool_arguments_are_rejected_not_forwarded():
    with pytest.raises(PrismError) as err:
        tr(
            build(
                messages=[
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "t1", "function": {"name": "p", "arguments": "{oops"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "t1", "content": "x"},
                ]
            )
        )
    assert err.value.kind is ErrorKind.INVALID_REQUEST


def test_tools_are_translated_to_input_schema():
    body, _ = tr(
        build(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Look up weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                        "strict": True,
                    },
                }
            ],
            tool_choice="required",
        )
    )
    assert body["tools"][0]["name"] == "get_weather"
    assert body["tools"][0]["input_schema"]["required"] == ["city"]
    assert body["tools"][0]["strict"] is True
    assert body["tool_choice"] == {"type": "any"}


def test_named_tool_choice_maps_to_tool_type():
    body, _ = tr(
        build(
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice={"type": "function", "function": {"name": "f"}},
        )
    )
    assert body["tool_choice"] == {"type": "tool", "name": "f"}


def test_tool_choice_none_omits_the_tool_list():
    body, warnings = tr(
        build(
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice="none",
        )
    )
    assert "tools" not in body
    assert "tools_omitted_for_tool_choice_none" in warnings


def test_parallel_tool_calls_false_disables_parallel_use():
    body, _ = tr(
        build(
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice="auto",
            parallel_tool_calls=False,
        )
    )
    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_trailing_assistant_message_is_rejected_as_a_prefill():
    with pytest.raises(PrismError) as err:
        tr(
            build(
                messages=[
                    {"role": "user", "content": "Write JSON"},
                    {"role": "assistant", "content": "{"},
                ]
            )
        )
    assert err.value.kind is ErrorKind.TRANSLATION
    assert err.value.status_code == 400


def test_n_greater_than_one_is_rejected_rather_than_silently_ignored():
    with pytest.raises(PrismError) as err:
        tr(build(n=3))
    assert err.value.param == "n"


def test_n_equal_to_one_is_accepted():
    body, _ = tr(build(n=1))
    assert body["max_tokens"] == 4096


def test_stop_string_becomes_a_stop_sequence_list():
    body, _ = tr(build(stop="END"))
    assert body["stop_sequences"] == ["END"]


def test_data_uri_image_becomes_a_base64_source():
    data = base64.b64encode(b"not-a-real-png").decode()
    body, _ = tr(
        build(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{data}"},
                        },
                    ],
                }
            ]
        )
    )
    blocks = body["messages"][0]["content"]
    assert blocks[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": data,
    }


def test_json_schema_response_format_becomes_output_config():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    body, _ = tr(
        build(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema},
            }
        )
    )
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": schema}


def test_json_object_response_format_is_flagged_as_unenforced():
    _, warnings = tr(build(response_format={"type": "json_object"}))
    assert "response_format_json_object_unenforced" in warnings


def test_effort_is_dropped_on_models_without_it():
    _, warnings = tr(build(model="claude-haiku-4-5", prism={"effort": "high"}), HAIKU)
    assert "effort_dropped_unsupported_model" in warnings


def test_effort_is_forwarded_where_supported():
    body, _ = tr(build(prism={"effort": "low"}))
    assert body["output_config"]["effort"] == "low"


def test_first_message_must_be_a_user_turn():
    with pytest.raises(PrismError):
        tr(
            build(
                messages=[{"role": "assistant", "content": "hi"}, {"role": "user", "content": "x"}]
            )
        )
