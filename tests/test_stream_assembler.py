"""The state machine over open content blocks.

These tests drive the assembler directly with upstream event sequences, because
the reassembly is where the real difficulty of the streaming translation lives.
"""

from __future__ import annotations

import json

from prism.translation.stream import StreamAssembler


def make(**kwargs) -> StreamAssembler:
    return StreamAssembler(model_requested="gpt-4o", completion_id="chatcmpl-x", **kwargs)


def feed(assembler: StreamAssembler, events: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for event in events:
        chunks.extend(assembler.handle(event))
    return chunks


def start(**usage) -> dict:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [],
            "usage": {"input_tokens": 10, "output_tokens": 1, **usage},
        },
    }


def deltas(chunks: list[dict]) -> list[dict]:
    return [c["choices"][0]["delta"] for c in chunks if c.get("choices")]


def test_text_deltas_are_relayed_and_accumulated():
    a = make()
    chunks = feed(
        a,
        [
            start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Par"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "is."},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ],
    )
    # Relayed as they arrived...
    assert [d.get("content") for d in deltas(chunks) if "content" in d] == ["", "Par", "is."]
    # ...and simultaneously buffered into the complete message.
    assert a.to_message()["content"] == [{"type": "text", "text": "Paris."}]
    assert a.complete


def test_the_first_chunk_announces_the_assistant_role():
    a = make()
    chunks = feed(a, [start()])
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[0]["object"] == "chat.completion.chunk"


def test_ping_and_unknown_events_are_ignored_not_fatal():
    a = make()
    # A gateway that raises on an event type added upstream after it shipped
    # breaks every client at once.
    assert a.handle({"type": "ping"}) == []
    assert a.handle({"type": "some_future_event", "index": 0}) == []


def test_tool_input_arrives_as_partial_json_and_is_parsed_only_at_block_stop():
    a = make()
    chunks = feed(
        a,
        [
            start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"ci'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": 'ty": "Par'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": 'is"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 9},
            },
            {"type": "message_stop"},
        ],
    )
    tool_deltas = [d for d in deltas(chunks) if "tool_calls" in d]
    # The opening chunk carries id and name; the rest carry argument fragments,
    # exactly as an OpenAI client expects to reassemble them.
    assert tool_deltas[0]["tool_calls"][0]["id"] == "toolu_01"
    assert tool_deltas[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    fragments = "".join(d["tool_calls"][0]["function"]["arguments"] for d in tool_deltas[1:])
    assert json.loads(fragments) == {"city": "Paris"}
    # And the buffered message holds it as a parsed object, not a string.
    assert a.to_message()["content"][0]["input"] == {"city": "Paris"}


def test_tool_call_index_counts_tool_calls_not_content_blocks():
    # A text block before the first tool_use would shift the block indices; the
    # OpenAI tool_calls index must not shift with them.
    a = make()
    chunks = feed(
        a,
        [
            start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Looking."},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
            },
            {"type": "content_block_stop", "index": 2},
        ],
    )
    indices = [d["tool_calls"][0]["index"] for d in deltas(chunks) if "tool_calls" in d]
    assert indices == [0, 1]


def test_truncated_tool_input_is_reported_not_guessed_at():
    a = make()
    feed(
        a,
        [
            start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city": "Par'},
            },
        ],
    )
    assert a.truncated_tool_input
    assert not a.complete
    # Half a JSON object is not a tool call; it must not be handed on as one.
    assert a.to_message()["content"][0]["input"] == {}


def test_thinking_deltas_go_to_reasoning_content():
    a = make()
    chunks = feed(
        a,
        [
            start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Weighing options."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "abc123"},
            },
            {"type": "content_block_stop", "index": 0},
        ],
    )
    assert [d["reasoning_content"] for d in deltas(chunks) if "reasoning_content" in d] == [
        "Weighing options."
    ]
    # The signature is an opaque multi-turn artefact: never relayed to the client.
    assert all("signature" not in d for d in deltas(chunks))


def test_thinking_policy_controls_what_reaches_the_trace():
    events = [
        start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "secret reasoning"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Paris."},
        },
        {"type": "content_block_stop", "index": 1},
    ]

    persisted = make(thinking_policy="persist")
    feed(persisted, events)
    assert persisted.to_message()["content"][0]["thinking"] == "secret reasoning"
    # Signatures are dropped even when thinking is persisted — large and useless
    # outside a follow-up request.
    assert "signature" not in persisted.to_message()["content"][0]

    redacted = make(thinking_policy="redact")
    feed(redacted, events)
    assert redacted.to_message()["content"][0]["thinking"] == "[redacted: 16 chars]"

    dropped = make(thinking_policy="drop")
    feed(dropped, events)
    assert [b["type"] for b in dropped.to_message()["content"]] == ["text"]


def test_blocks_are_emitted_in_index_order_regardless_of_arrival_order():
    a = make()
    feed(
        a,
        [
            start(),
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": "second"},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "first"},
            },
            {"type": "content_block_stop", "index": 0},
        ],
    )
    assert [b["text"] for b in a.to_message()["content"]] == ["first", "second"]


def test_usage_merges_message_start_input_with_message_delta_output():
    a = make()
    feed(
        a,
        [
            start(cache_read_input_tokens=900, cache_creation_input_tokens=50),
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 77},
            },
            {"type": "message_stop"},
        ],
    )
    usage = a.token_usage()
    assert usage.input_tokens == 10
    assert usage.cache_read_input_tokens == 900
    assert usage.cache_creation_input_tokens == 50
    # The running total on message_start is superseded, not added to.
    assert usage.output_tokens == 77


def test_an_unfinished_stream_is_never_marked_complete():
    a = make()
    feed(
        a,
        [
            start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "half an ans"},
            },
        ],
    )
    # A partial response is not a cheap cache entry, it is a wrong one.
    assert not a.complete
    assert a.to_message()["content"] == [{"type": "text", "text": "half an ans"}]


def test_the_reconstructed_message_feeds_the_ordinary_translation_path():
    # The point of rebuilding the Anthropic message rather than assembling an
    # OpenAI response directly: one translation path, no second implementation
    # that can drift from the non-streaming one.
    from prism.translation.response import translate

    a = make()
    feed(
        a,
        [
            start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Paris."},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ],
    )
    completion = translate(a.to_message(), model_requested="gpt-4o")
    assert completion.choices[0].message.content == "Paris."
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage.completion_tokens == 5
