"""The layer that can be wrong: scoping, thresholds, and calibration."""

from __future__ import annotations

from pathlib import Path

import pytest

from prism.caching.calibration import (
    LabelledPair,
    ThresholdPoint,
    calibrate,
    choose_operating_point,
    load_pairs,
    sweep,
)
from prism.caching.embeddings import HashingEmbedder, cosine
from prism.caching.keys import CacheScope, exact_key, query_text
from prism.caching.semantic import (
    CacheHit,
    CacheMiss,
    SemanticCache,
    SemanticCacheConfig,
    UnsafeEmbedderError,
)
from prism.caching.store import CacheEntry, InMemoryStore

PAIRS = Path(__file__).resolve().parents[1] / "datasets" / "cache_pairs.jsonl"


def body(text: str, **kwargs):
    out = {
        "model": "claude-opus-5",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }
    out.update(kwargs)
    return out


def build_cache(**config) -> SemanticCache:
    return SemanticCache(
        InMemoryStore(),
        HashingEmbedder(),
        SemanticCacheConfig(enabled=True, **config),
        # The stub is not semantically meaningful; these tests exercise the
        # plumbing around the threshold, not retrieval quality.
        allow_unsafe_embedder=True,
    )


# --- the isolation that no threshold can provide --------------------------


async def test_a_tenant_can_never_hit_another_tenants_entry():
    """The single most important property in this layer.

    If tenant A's cached response reaches tenant B, that is a data leak, and no
    similarity threshold prevents it — the queries are *identical*, so similarity
    is 1.0. Only the scope partition stops it.
    """
    cache = build_cache(threshold=0.5)
    question = "What is the standard retention period for our audit logs?"

    a = CacheScope.from_request(body(question), tenant_id="tenant-a")
    b = CacheScope.from_request(body(question), tenant_id="tenant-b")
    assert a.key != b.key

    await cache.put(body(question), a, {"secret": "tenant A internal answer"})

    hit = await cache.lookup(body(question), a)
    assert isinstance(hit, CacheHit)

    # Same question, byte for byte. Similarity is 1.0 and it must still miss.
    leaked = await cache.lookup(body(question), b)
    assert isinstance(leaked, CacheMiss)
    assert leaked.reason == "scope is empty"


async def test_a_hit_across_model_tiers_is_impossible():
    # Haiku and Opus answer differently. A cross-tier hit would serve the cheap
    # answer to someone who paid for the expensive one, and would wreck the
    # router's measurements in week 9.
    cache = build_cache(threshold=0.5)
    question = "Explain the tradeoff between latency and throughput in one paragraph."

    opus = CacheScope.from_request(body(question, model="claude-opus-5"), tenant_id="t1")
    haiku = CacheScope.from_request(body(question, model="claude-haiku-4-5"), tenant_id="t1")
    await cache.put(body(question), opus, {"from": "opus"})

    assert isinstance(await cache.lookup(body(question), haiku), CacheMiss)


def test_every_scope_component_changes_the_partition():
    base = CacheScope.from_request(body("a question about caching"), tenant_id="t1")
    variants = {
        "tenant": CacheScope.from_request(body("a question about caching"), tenant_id="t2"),
        "model": CacheScope.from_request(
            body("a question about caching", model="claude-sonnet-5"), tenant_id="t1"
        ),
        "temperature": CacheScope.from_request(
            body("a question about caching", temperature=0.9), tenant_id="t1"
        ),
        "system": CacheScope.from_request(
            body("a question about caching", system="You are terse."), tenant_id="t1"
        ),
        "tools": CacheScope.from_request(
            body("a question about caching", tools=[{"name": "f", "input_schema": {}}]),
            tenant_id="t1",
        ),
    }
    for name, variant in variants.items():
        assert variant.key != base.key, f"{name} did not change the scope key"


def test_a_prompt_version_bump_changes_the_scope():
    # This is why a version bump has a predictable cache cost rather than a
    # mysterious hit-rate cliff.
    text = "a question about caching"
    v1 = CacheScope.from_request(body(text), tenant_id="t1", prompt_version="assistant@v1")
    v2 = CacheScope.from_request(body(text), tenant_id="t1", prompt_version="assistant@v2")
    assert v1.key != v2.key


def test_tool_order_does_not_change_the_scope_key():
    # An unsorted json.dumps would hash an identical tool set differently between
    # runs and silently empty the cache.
    tools_a = [{"name": "a", "input_schema": {"x": 1, "y": 2}}]
    tools_b = [{"name": "a", "input_schema": {"y": 2, "x": 1}}]
    left = CacheScope.from_request(body("q", tools=tools_a), tenant_id="t")
    right = CacheScope.from_request(body("q", tools=tools_b), tenant_id="t")
    assert left.key == right.key


# --- what gets embedded ---------------------------------------------------


def test_only_the_final_user_turn_is_embedded():
    # Embedding the whole conversation makes every long thread a unique vector
    # and drives the hit rate to zero.
    payload = body("the actual question")
    payload["messages"] = [
        {"role": "user", "content": [{"type": "text", "text": "earlier turn"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "earlier reply"}]},
        {"role": "user", "content": [{"type": "text", "text": "the actual question"}]},
    ]
    assert query_text(payload) == "the actual question"


def test_the_exact_key_covers_the_whole_message_list():
    a = body("same")
    b = body("same")
    scope = CacheScope.from_request(a, tenant_id="t")
    assert exact_key(a, scope) == exact_key(b, scope)

    c = body("different")
    assert exact_key(c, scope) != exact_key(a, scope)


# --- refusing to be unsafe ------------------------------------------------


def test_the_cache_refuses_a_non_production_embedder():
    # "Nearest" under a hashing stub is not semantically near, so a hit would be
    # an arbitrary wrong answer.
    with pytest.raises(UnsafeEmbedderError, match="not production grade"):
        SemanticCache(InMemoryStore(), HashingEmbedder(), SemanticCacheConfig(enabled=True))


async def test_the_cache_is_off_by_default():
    # Nothing serves traffic until a threshold has been calibrated.
    cache = SemanticCache(
        InMemoryStore(),
        HashingEmbedder(),
        SemanticCacheConfig(),
        allow_unsafe_embedder=True,
    )
    result = await cache.lookup(body("anything at all, reasonably long"), CacheScope("t", "m"))
    assert isinstance(result, CacheMiss)
    assert result.reason == "semantic cache disabled"


async def test_short_queries_are_refused():
    # Few tokens to disambiguate means high cosine similarity between different
    # questions, which is where false hits cluster.
    cache = build_cache(threshold=0.1, min_query_chars=24)
    scope = CacheScope.from_request(body("hi"), tenant_id="t")
    await cache.put(body("hi"), scope, {"answer": "hello"})
    result = await cache.lookup(body("hi"), scope)
    assert isinstance(result, CacheMiss)
    assert "shorter than" in result.reason


async def test_a_truncated_stream_is_never_stored():
    # A partial answer is a wrong cache entry, not a cheap one.
    cache = build_cache(threshold=0.5)
    scope = CacheScope.from_request(body("a long enough question to be cached"), tenant_id="t")
    stored = await cache.put(
        body("a long enough question to be cached"), scope, {"partial": True}, cacheable=False
    )
    assert stored is False
    assert await cache.store.size() == 0


# --- the threshold --------------------------------------------------------


async def test_a_miss_records_how_close_it_came():
    # Production misses are what the threshold sweep re-tunes on, so the near
    # score is kept rather than discarded.
    cache = build_cache(threshold=0.999)
    scope = CacheScope.from_request(body("what is the capital of France"), tenant_id="t")
    await cache.put(body("what is the capital of France"), scope, {"a": 1})
    result = await cache.lookup(body("what is the capital of Germany"), scope)
    assert isinstance(result, CacheMiss)
    assert result.best_similarity is not None
    assert result.nearest_query == "what is the capital of France"


async def test_raising_the_threshold_turns_a_hit_into_a_miss():
    question = "How do I reverse a list in Python, step by step?"
    near = "How do I reverse a list in Python, step by step please?"

    permissive = build_cache(threshold=0.5)
    scope = CacheScope.from_request(body(question), tenant_id="t")
    await permissive.put(body(question), scope, {"a": 1})
    assert isinstance(await permissive.lookup(body(near), scope), CacheHit)

    strict = build_cache(threshold=0.9999)
    await strict.put(body(question), scope, {"a": 1})
    assert isinstance(await strict.lookup(body(near), scope), CacheMiss)


# --- calibration ----------------------------------------------------------


def test_the_shipped_pair_set_loads_and_has_both_classes():
    pairs = load_pairs(PAIRS)
    assert len(pairs) >= 40
    assert any(p.equivalent for p in pairs)
    assert any(not p.equivalent for p in pairs)


def test_the_pair_set_contains_hard_negatives():
    # A near-identical string with a different correct answer is the pair a
    # threshold tuned on hit rate gets wrong. Without these the curve is easy
    # and the chosen threshold is worthless.
    pairs = load_pairs(PAIRS)
    negatives = [p for p in pairs if not p.equivalent]
    embedder = HashingEmbedder()
    scored = []
    for pair in negatives:
        a = embedder.encode([pair.query_a], is_query=True)[0]
        b = embedder.encode([pair.query_b], is_query=False)[0]
        scored.append(cosine(a, b))
    # At least a few non-equivalent pairs must be *very* close.
    assert max(scored) > 0.8


def test_the_false_hit_rate_is_the_share_of_served_responses_that_were_wrong():
    point = ThresholdPoint(
        threshold=0.9, true_positives=90, false_positives=10, true_negatives=100, false_negatives=50
    )
    assert point.false_hit_rate == pytest.approx(0.10)  # 10 of 100 served
    assert point.precision == pytest.approx(0.90)
    assert point.recall == pytest.approx(90 / 140)
    # Hit rate looks healthy while one in ten answers is wrong, which is why it
    # is not the number to report.
    assert point.hit_rate == pytest.approx(100 / 250)


def test_zero_observed_false_hits_does_not_mean_zero_risk():
    point = ThresholdPoint(
        threshold=0.99, true_positives=20, false_positives=0, true_negatives=100, false_negatives=80
    )
    assert point.false_hit_rate == 0.0
    # 0/20 puts the upper bound near 16%, not at zero.
    assert point.false_hit_interval().high > 0.10


def test_the_operating_point_is_chosen_on_the_upper_bound_not_the_estimate():
    curve = [
        # Observed 0/8, but the interval is wide: not safe at a 1% ceiling.
        ThresholdPoint(
            0.90, true_positives=8, false_positives=0, true_negatives=50, false_negatives=10
        ),
    ]
    assert choose_operating_point(curve, max_false_hit_rate=0.01) is None
    # On the point estimate alone it would have been accepted.
    assert choose_operating_point(curve, max_false_hit_rate=0.01, use_upper_bound=False) is not None


def test_a_threshold_that_serves_almost_nothing_is_rejected():
    # With two hits and no misses the observed rate is meaningless.
    curve = [
        ThresholdPoint(
            0.999, true_positives=2, false_positives=0, true_negatives=200, false_negatives=50
        )
    ]
    assert choose_operating_point(curve, max_false_hit_rate=0.5, min_served=5) is None


def test_no_viable_threshold_is_a_real_answer():
    # A cache that cannot be made safe on this corpus should stay off, and the
    # report has to be able to say so rather than picking the least-bad point.
    pairs = [LabelledPair("identical text", "identical text", equivalent=False)] * 20
    result = calibrate(pairs, HashingEmbedder(), max_false_hit_rate=0.01)
    assert result.chosen is None
    assert "NO OPERATING POINT" in result.render()


def test_the_curve_is_monotonic_in_the_right_directions():
    pairs = load_pairs(PAIRS)
    curve = sweep(pairs, HashingEmbedder())
    hit_rates = [p.hit_rate for p in curve]
    recalls = [p.recall for p in curve]
    # Raising the threshold can only serve fewer requests and recall less.
    assert hit_rates == sorted(hit_rates, reverse=True)
    assert recalls == sorted(recalls, reverse=True)


def test_auc_is_reported_because_it_is_threshold_independent():
    # A low AUC means no operating point is good, and no tuning will fix it.
    result = calibrate(load_pairs(PAIRS), HashingEmbedder())
    assert 0.0 <= result.auc <= 1.0


def test_the_report_states_the_exposure_not_just_the_hit_rate():
    curve = [
        ThresholdPoint(
            0.95, true_positives=60, false_positives=0, true_negatives=100, false_negatives=20
        )
    ]
    from prism.caching.calibration import CalibrationResult

    result = CalibrationResult(
        curve=curve,
        chosen=curve[0],
        max_false_hit_rate=0.05,
        n_pairs=180,
        embedder="stub",
    )
    rendered = result.render()
    assert "false_hit" in rendered
    assert "95% CI upper bound" in rendered


# --- the store ------------------------------------------------------------


async def test_the_in_memory_store_partitions_by_scope():
    store = InMemoryStore()
    await store.put(CacheEntry("scope-a", "q", [1.0, 0.0], {"a": 1}))
    await store.put(CacheEntry("scope-b", "q", [1.0, 0.0], {"b": 2}))
    assert await store.size() == 2

    found = await store.nearest(CacheScope("t", "m"), [1.0, 0.0])
    # A scope with no entries returns nothing, even though vectors exist.
    assert found == []
