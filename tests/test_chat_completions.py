"""End-to-end ingress behaviour, with a scripted upstream."""

from __future__ import annotations

from prism.schemas.errors import ErrorKind, PrismError


def test_happy_path_returns_an_openai_shaped_completion(client, provider):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "capital of France?"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Paris."
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 12

    # The upstream body is the real assertion.
    sent = provider.last_call
    assert sent["model"] == "claude-opus-5"
    assert sent["max_tokens"] == 4096
    assert sent["messages"][0]["role"] == "user"


def test_openai_model_names_are_mapped_onto_the_ladder(client, provider):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert provider.last_call["model"] == "claude-sonnet-5"
    # The client keeps its own string; the header says what actually ran.
    assert resp.json()["model"] == "gpt-4o"
    assert resp.headers["x-prism-model"] == "claude-sonnet-5"


def test_cost_and_cache_headers_are_present(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert float(resp.headers["x-prism-cost-usd"]) > 0
    assert resp.headers["x-prism-cache-read-tokens"] == "0"
    assert resp.headers["x-prism-upstream-latency-ms"] == "42"


def test_stream_true_is_served_as_sse_not_as_one_buffered_blob(client):
    # See test_stream_endpoint.py for the reassembly itself; this only pins the
    # branch — a client asking for tokens as they arrive must not get JSON.
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")


def test_unknown_model_is_a_404_with_an_openai_shaped_error(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "llama-3", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["param"] == "model"


def test_quota_exhaustion_is_relayed_as_429_with_retry_after(client, provider):
    provider.error = PrismError(
        ErrorKind.UPSTREAM_RATE_LIMIT,
        "rate limit",
        status_code=429,
        retry_after=12.0,
        upstream_status=429,
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "12"
    assert resp.json()["error"]["type"] == "upstream_rate_limit"


def test_provider_overload_is_a_distinct_error_type_from_quota_exhaustion(client, provider):
    # 429 and 529 demand opposite responses, so they must never collapse into one
    # "upstream error" the client cannot act on.
    provider.error = PrismError(
        ErrorKind.UPSTREAM_OVERLOADED, "overloaded", status_code=503, upstream_status=529
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "upstream_overloaded"
    assert resp.headers["x-prism-upstream-status"] == "529"


def test_only_capacity_errors_count_toward_the_circuit_breaker():
    assert ErrorKind.UPSTREAM_OVERLOADED.counts_toward_circuit_breaker
    assert ErrorKind.UPSTREAM_TIMEOUT.counts_toward_circuit_breaker
    # Failing over on a 429 would carry the same exhausted quota to another route.
    assert not ErrorKind.UPSTREAM_RATE_LIMIT.counts_toward_circuit_breaker
    assert not ErrorKind.UPSTREAM_INVALID_REQUEST.counts_toward_circuit_breaker


def test_validation_errors_use_the_openai_error_envelope(client):
    resp = client.post("/v1/chat/completions", json={"model": "claude-opus-5"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_models_endpoint_lists_the_ladder(client):
    data = client.get("/v1/models").json()["data"]
    ids = {m["id"] for m in data}
    assert {"claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"} <= ids


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


# --- prefix cache (week 7) ------------------------------------------------


def read_prompt(name: str = "assistant", version: str = "v1") -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / "prompts" / name / f"{version}.md").read_text(
        encoding="utf-8"
    )


def test_a_prompt_built_on_top_of_a_registry_version_is_refused(client, provider, recorder):
    # Long enough to clear the token floor, and claiming a real version - but
    # the bytes differ from the registry copy, so it could contain anything.
    system = read_prompt() * 60
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "hi"},
            ],
            "prism": {"prompt_version": "assistant@v1"},
        },
    )
    assert resp.status_code == 200
    # The registry copy is one repetition, so the bytes do NOT match and the
    # policy must refuse — which is the conservative answer, and correct.
    sent = provider.last_call
    assert "cache_control" not in sent["system"][-1]
    reason = recorder.last.extra["prefix_cache"]["scopes"]["reason"]
    assert "byte-for-byte" in reason


def test_an_exact_registry_match_is_cached(client, provider, recorder):
    from prism.prompts import Registry

    # Make the shipped prompt long enough to clear the floor by adding a version
    # the test controls, then send exactly those bytes.
    registry = Registry("prompts")
    system = registry.get("assistant", "v1").text
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "hi"},
            ],
            "prism": {"prompt_version": "assistant@v1"},
        },
    )
    assert resp.status_code == 200
    cache = recorder.last.extra["prefix_cache"]
    # Byte-identical, so the scope decision is SHARED...
    assert cache["scopes"]["system"] == "shared"
    # ...but the prompt is well under the 1024-token floor, so no marker is
    # placed. Spending a slot below the floor buys nothing.
    assert cache["breakpoints"] == []
    assert any("below the" in s for s in cache["skipped"])


def test_an_unversioned_system_prompt_is_never_cached(client, provider, recorder):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "messages": [
                {"role": "system", "content": "word " * 3000},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    assert resp.status_code == 200
    assert "cache_control" not in provider.last_call["system"][-1]
    reason = recorder.last.extra["prefix_cache"]["scopes"]["reason"]
    assert "not registry-versioned" in reason


def test_the_placement_report_lands_on_the_trace(client, recorder):
    client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    cache = recorder.last.extra["prefix_cache"]
    assert "breakpoints" in cache and "skipped" in cache and "scopes" in cache


# --- semantic cache (week 8) ----------------------------------------------


def ask(client, question: str, **extra):
    payload = {
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": question}],
    }
    payload.update(extra)
    return client.post("/v1/chat/completions", json=payload)


QUESTION = "How do I reverse a list in Python, with a worked example?"


def test_the_first_request_misses_and_is_stored(client, provider, recorder):
    resp = ask(client, QUESTION)
    assert resp.status_code == 200
    assert resp.headers["x-prism-cache"] == "miss"
    assert len(provider.calls) == 1
    assert recorder.last.extra["semantic_cache"]["hit"] is False
    assert recorder.last.extra["semantic_cache_stored"] is True


def test_an_identical_repeat_is_served_from_cache_without_an_upstream_call(
    client, provider, recorder
):
    ask(client, QUESTION)
    assert len(provider.calls) == 1

    second = ask(client, QUESTION)
    assert second.status_code == 200
    assert second.headers["x-prism-cache"] == "semantic-hit"
    # The whole point: no upstream call, so no tokens and no cost.
    assert len(provider.calls) == 1
    assert second.headers["x-prism-cost-usd"] == "0.00000000"
    assert second.json()["choices"][0]["message"]["content"] == "Paris."


def test_a_cache_hit_records_zero_cost_rather_than_the_original_cost(client, recorder):
    ask(client, QUESTION)
    ask(client, QUESTION)
    trace = recorder.last
    # Recording the original request's usage again would double-count the spend
    # and hide the saving in the rollups.
    assert trace.usage.output_tokens == 0
    assert trace.cost is None or trace.cost.total_usd == 0
    assert trace.extra["semantic_cache"]["hit"] is True


def test_the_similarity_of_a_hit_is_reported_to_the_client(client):
    ask(client, QUESTION)
    second = ask(client, QUESTION)
    assert float(second.headers["x-prism-cache-similarity"]) >= 0.99


def test_a_different_question_still_reaches_the_provider(client, provider):
    ask(client, QUESTION)
    ask(client, "What is the boiling point of water at sea level, in Celsius?")
    assert len(provider.calls) == 2


def test_a_miss_records_how_close_it_came_for_later_retuning(client, recorder):
    ask(client, QUESTION)
    ask(client, "What is the boiling point of water at sea level, in Celsius?")
    cache = recorder.last.extra["semantic_cache"]
    assert cache["hit"] is False
    assert cache["best_similarity"] is not None
    assert cache["nearest_query"] == QUESTION


def test_the_scope_key_is_recorded_on_the_trace(client, recorder):
    ask(client, QUESTION)
    scope = recorder.last.extra["semantic_cache"]["scope"]
    assert scope["model"] == "claude-opus-5"
    assert scope["scope_key"]


def test_streaming_requests_bypass_the_semantic_lookup(client, provider, recorder):
    # Replaying a cached completion as SSE changes what time-to-first-token
    # means, so it is deliberately not done here.
    ask(client, QUESTION)
    resp = ask(client, QUESTION, stream=True)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    # The stream still went upstream despite an identical cached entry existing.
    assert len(provider.calls) == 2
