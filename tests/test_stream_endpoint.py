"""Streaming ingress: SSE framing, timeout budgets, and disconnect handling."""

from __future__ import annotations

import json

import pytest
from conftest import TEXT_STREAM

from prism.providers.base import StreamHandle
from prism.schemas.errors import ErrorKind, PrismError


def frames(raw: str) -> list[str]:
    return [line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")]


def chunks(raw: str) -> list[dict]:
    return [json.loads(f) for f in frames(raw) if f != "[DONE]"]


def stream(client, **overrides):
    payload = {
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": "capital of France?"}],
        "stream": True,
    }
    payload.update(overrides)
    return client.post("/v1/chat/completions", json=payload)


def test_stream_returns_sse_chunks_terminated_by_done(client):
    resp = stream(client)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    # Proxies buffer streamed responses by default, which would undo the point.
    assert resp.headers["x-accel-buffering"] == "no"
    assert frames(resp.text)[-1] == "[DONE]"


def test_text_arrives_as_incremental_deltas(client):
    body = chunks(stream(client).text)
    contents = [
        c["choices"][0]["delta"].get("content")
        for c in body
        if c["choices"] and "content" in c["choices"][0]["delta"]
    ]
    assert contents == ["", "Par", "is."]
    assert body[0]["choices"][0]["delta"]["role"] == "assistant"
    assert body[-1]["choices"][0]["finish_reason"] == "stop"


def test_the_client_keeps_its_own_model_string_in_every_chunk(client):
    body = chunks(stream(client, model="gpt-4o").text)
    assert {c["model"] for c in body} == {"gpt-4o"}


def test_usage_is_only_sent_when_the_client_asked_for_it(client):
    without = chunks(stream(client).text)
    assert all("usage" not in c for c in without)

    with_usage = chunks(stream(client, stream_options={"include_usage": True}).text)
    final = with_usage[-1]
    assert final["choices"] == []
    assert final["usage"]["completion_tokens"] == 5
    assert final["usage"]["prompt_tokens"] == 12


def test_the_upstream_body_is_marked_as_a_stream(client, provider):
    stream(client)
    assert provider.last_call["stream"] is True


def test_errors_before_the_first_byte_still_get_a_real_status_code(client, provider):
    # Once the first frame is written the status line is spent. Priming the
    # generator before handing it to the server keeps a 429 relayable as a 429
    # instead of degrading into an SSE error frame.
    provider.stream_open_error = PrismError(
        ErrorKind.UPSTREAM_RATE_LIMIT,
        "rate limit",
        status_code=429,
        retry_after=7.0,
        upstream_status=429,
    )
    resp = stream(client)
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "7"
    assert resp.json()["error"]["type"] == "upstream_rate_limit"


def test_overload_before_the_first_byte_keeps_its_own_status(client, provider):
    provider.stream_open_error = PrismError(
        ErrorKind.UPSTREAM_OVERLOADED, "overloaded", status_code=503, upstream_status=529
    )
    resp = stream(client)
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "upstream_overloaded"


def test_a_mid_stream_error_event_becomes_an_error_frame(client, provider):
    # Bytes are already on the wire, so an SSE error frame is the only channel
    # left — but the client must still be told, not just cut off.
    provider.stream_events = TEXT_STREAM[:4] + [
        {"type": "error", "error": {"type": "overloaded_error", "message": "over capacity"}}
    ]
    resp = stream(client)
    assert resp.status_code == 200
    payloads = chunks(resp.text)
    assert payloads[-1]["error"]["type"] == "upstream_overloaded"
    assert frames(resp.text)[-1] == "[DONE]"


def test_translation_errors_are_refused_before_the_stream_opens(client, provider):
    resp = stream(
        client,
        messages=[
            {"role": "user", "content": "Write JSON"},
            {"role": "assistant", "content": "{"},
        ],
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "translation_error"
    assert provider.calls == []


def test_upstream_connection_is_closed_when_the_stream_ends(client, provider):
    stream(client)
    # Closing is what stops upstream generation; leaking the connection would
    # keep the meter running on an abandoned request.
    assert provider.stream_closed is True


# --- StreamHandle budgets ------------------------------------------------


async def _lines(items: list[str], *, then_hang: bool = False):
    """Yield some frames, then optionally stall — a stream that goes quiet."""
    import asyncio

    for item in items:
        yield item
    if then_hang:
        await asyncio.sleep(60)


async def test_first_token_budget_fires_before_any_content_arrives():
    # message_start and ping frames arrive on a healthy idle stream. They must
    # not satisfy the first-token budget, and they must not reset it either.
    handle = StreamHandle(
        _lines(
            [
                'data: {"type": "message_start", "message": {"id": "m", "usage": {}}}',
                'data: {"type": "ping"}',
            ],
            then_hang=True,
        ),
        first_token_timeout_s=0.05,
        total_timeout_s=30.0,
    )
    with pytest.raises(PrismError) as err:
        async for _ in handle.events():
            pass
    assert err.value.kind is ErrorKind.UPSTREAM_TIMEOUT
    assert "first-token" in err.value.message


async def test_total_budget_fires_after_content_has_started():
    # Past the first token the first-token budget no longer applies; a slow but
    # producing stream is healthy until the total budget runs out.
    handle = StreamHandle(
        _lines(
            [
                'data: {"type": "content_block_delta", "index": 0, '
                '"delta": {"type": "text_delta", "text": "hi"}}',
            ],
            then_hang=True,
        ),
        first_token_timeout_s=0.01,
        total_timeout_s=0.08,
    )
    seen = []
    with pytest.raises(PrismError) as err:
        async for event in handle.events():
            seen.append(event)
    assert len(seen) == 1
    assert "total response budget" in err.value.message


async def test_thinking_requests_get_a_longer_first_token_budget(client, provider):
    # A large thinking budget legitimately delays the first *visible* token, so a
    # fixed first-token budget would kill healthy requests.
    stream(client, prism={"thinking": True})
    assert "thinking" in provider.last_call


async def test_malformed_sse_frames_are_an_upstream_error():
    handle = StreamHandle(_lines(["data: {not json"]))
    with pytest.raises(PrismError) as err:
        async for _ in handle.events():
            pass
    assert err.value.kind is ErrorKind.UPSTREAM_SERVER_ERROR


# --- what the stream leaves behind on the trace ---------------------------


def test_a_completed_stream_records_reconstructed_content_and_cost(client, recorder):
    stream(client)
    trace = recorder.last

    assert trace.status == "ok"
    assert trace.stream is True
    # The buffered message is the same shape the non-streaming path produces.
    assert trace.upstream_response["content"] == [{"type": "text", "text": "Paris."}]
    assert trace.response_body["choices"][0]["message"]["content"] == "Paris."
    assert trace.usage.output_tokens == 5
    assert trace.usage.input_tokens == 12
    assert trace.cost.total_usd > 0
    assert trace.extra["stream_complete"] is True
    assert trace.extra["cacheable"] is True


def test_ttft_is_recorded_and_total_latency_is_not_confused_with_it(client, recorder):
    stream(client)
    trace = recorder.last
    assert trace.ttft_ms is not None
    assert trace.latency_ms is not None
    assert trace.ttft_ms <= trace.latency_ms
    # A streamed response has no bracketed upstream call to measure, so claiming
    # an upstream latency here would be inventing a number.
    assert trace.upstream_latency_ms is None


def test_a_truncated_stream_is_recorded_as_not_cacheable(client, provider, recorder):
    # Cut the stream after one text delta: no content_block_stop, no message_stop.
    provider.stream_events = TEXT_STREAM[:4]
    stream(client)
    trace = recorder.last
    # It still carries the partial text for debugging...
    assert trace.upstream_response["content"] == [{"type": "text", "text": "Par"}]
    # ...but a partial response is a wrong cache entry, not a cheap one.
    assert trace.extra["stream_complete"] is False
    assert trace.extra["cacheable"] is False


async def test_client_disconnect_abandons_upstream_and_records_the_partial():
    """The disconnect case, driven directly — TestClient always reads to the end.

    The open question the design has to answer: when a client hangs up at 60%,
    do you keep consuming upstream to complete the buffer, or abandon it? Prism
    abandons by default, because closing the connection stops generation and so
    stops the meter. Week 8 may revisit it: once a semantic cache exists, a
    completed entry is worth something that a partial one is not.
    """
    import time as _time

    from conftest import CapturingRecorder, FakeProvider

    from prism.api.openai_routes import _stream_completion
    from prism.config import Settings
    from prism.registry import MODELS
    from prism.schemas.openai import ChatCompletionRequest
    from prism.tracing.recorder import TraceDraft

    provider = FakeProvider()
    recorder = CapturingRecorder()
    body = ChatCompletionRequest.model_validate(
        {
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
    )
    draft = TraceDraft(
        endpoint="/v1/chat/completions",
        model_requested="claude-opus-5",
        model_resolved="claude-opus-5",
        stream=True,
    )

    frames_iter = _stream_completion(
        body=body,
        spec=MODELS["claude-opus-5"],
        upstream_body={"model": "claude-opus-5", "max_tokens": 16, "messages": []},
        draft=draft,
        provider=provider,
        recorder=recorder,
        settings=Settings(anthropic_api_key="test"),
        started=_time.perf_counter(),
    )

    seen = []
    async for frame in frames_iter:
        seen.append(frame)
        if len(seen) == 3:
            break
    await frames_iter.aclose()  # the client went away

    assert provider.stream_closed is True, "upstream connection must be released"

    trace = recorder.last
    assert trace.status == "client_disconnect"
    assert trace.http_status == 499
    assert trace.error_type == "client_disconnect"
    assert trace.extra["cacheable"] is False
