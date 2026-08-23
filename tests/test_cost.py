"""Cost accounting across the five token classes."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from prism.cost import TokenUsage, compute_cost, naive_cost
from prism.registry import MODELS

OPUS = MODELS["claude-opus-5"]
SONNET = MODELS["claude-sonnet-5"]


def test_uncached_input_and_output_price_at_the_list_rate():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = compute_cost(usage, OPUS)
    assert cost.uncached_input_usd == Decimal("5.00")
    assert cost.output_usd == Decimal("25.00")


def test_cache_reads_price_at_a_tenth_and_writes_at_a_premium():
    reads = compute_cost(TokenUsage(cache_read_input_tokens=1_000_000), OPUS)
    assert reads.cached_input_usd == Decimal("0.500")

    writes = compute_cost(TokenUsage(cache_creation_input_tokens=1_000_000), OPUS)
    assert writes.cache_write_usd == Decimal("6.250")

    long_ttl = compute_cost(TokenUsage(cache_creation_input_tokens=1_000_000), OPUS, cache_ttl="1h")
    assert long_ttl.cache_write_usd == Decimal("10.0")


def test_the_naive_model_overstates_a_cache_heavy_request():
    # 90% prefix hit rate: the naive model bills every input token at list price.
    usage = TokenUsage(input_tokens=100_000, cache_read_input_tokens=900_000, output_tokens=1_000)
    real = compute_cost(usage, OPUS).total_usd
    naive = naive_cost(usage, OPUS)
    assert naive > real * 2
    # Concretely: $5.025 naive vs $0.975 real, a 5.2x overstatement.
    assert naive.quantize(Decimal("0.001")) == Decimal("5.025")
    assert real.quantize(Decimal("0.001")) == Decimal("0.975")


def test_the_naive_model_understates_a_write_heavy_request():
    usage = TokenUsage(cache_creation_input_tokens=1_000_000)
    assert compute_cost(usage, OPUS).total_usd > naive_cost(usage, OPUS)


def test_thinking_tokens_are_not_billed_twice():
    # They are already inside output_tokens; adding them again would inflate cost.
    plain = compute_cost(TokenUsage(output_tokens=1000), OPUS).total_usd
    with_thinking = compute_cost(
        TokenUsage(output_tokens=1000, thinking_tokens=800), OPUS
    ).total_usd
    assert plain == with_thinking


def test_introductory_pricing_applies_only_inside_its_window():
    usage = TokenUsage(input_tokens=1_000_000)
    inside = compute_cost(usage, SONNET, on=date(2026, 8, 22)).total_usd
    outside = compute_cost(usage, SONNET, on=date(2026, 9, 1)).total_usd
    assert inside == Decimal("2.00")
    assert outside == Decimal("3.00")


def test_breakdown_serializes_without_float_rounding():
    cost = compute_cost(TokenUsage(input_tokens=333_333), OPUS)
    payload = cost.as_json()
    assert isinstance(payload["total_usd"], str)
    assert Decimal(payload["total_usd"]) == cost.total_usd
