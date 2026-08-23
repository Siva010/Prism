"""Token estimation and the scope decision that gates breakpoint placement."""

from __future__ import annotations

import json

import pytest

from prism.caching.breakpoints import Scope
from prism.caching.policy import apply_to_request, decide_scopes
from prism.prompts import Registry
from prism.tokens import (
    EstimatorReport,
    EstimatorSample,
    calibrate,
    estimate_request,
    estimate_text,
)

LONG = "word " * 2000


@pytest.fixture
def registry(tmp_path) -> Registry:
    (tmp_path / "assistant").mkdir()
    (tmp_path / "assistant" / "v1.md").write_text(LONG, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"assistant": {"active": "v1"}}), encoding="utf-8"
    )
    return Registry(tmp_path)


def body(system=None, **kwargs):
    out = {
        "model": "claude-opus-5",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    if system is not None:
        out["system"] = system
    out.update(kwargs)
    return out


# --- scope decisions ------------------------------------------------------


def test_a_registry_matched_system_prompt_is_shared(registry: Registry):
    decision = decide_scopes(
        body(system=[{"type": "text", "text": LONG}]),
        prompt_version="assistant@v1",
        registry=registry,
    )
    assert decision.system is Scope.SHARED
    assert "byte-identical" in decision.reason


def test_a_modified_system_prompt_is_not_shared_even_with_a_valid_version(
    registry: Registry,
):
    # The claimed version exists, but the caller appended something per request.
    # Caching that under a shared key would leak whatever they added.
    decision = decide_scopes(
        body(system=[{"type": "text", "text": LONG + "\nTenant: Acme Corp."}]),
        prompt_version="assistant@v1",
        registry=registry,
    )
    assert decision.system is Scope.TENANT
    assert "byte-for-byte" in decision.reason


def test_an_unversioned_system_prompt_is_assumed_tenant_specific(registry: Registry):
    decision = decide_scopes(
        body(system=[{"type": "text", "text": LONG}]),
        prompt_version=None,
        registry=registry,
    )
    assert decision.system is Scope.TENANT
    assert "not registry-versioned" in decision.reason


def test_a_version_that_does_not_resolve_is_not_shared(registry: Registry):
    decision = decide_scopes(
        body(system=[{"type": "text", "text": LONG}]),
        prompt_version="assistant@v99",
        registry=registry,
    )
    assert decision.system is Scope.TENANT
    assert "did not resolve" in decision.reason


def test_conversation_history_is_never_shared(registry: Registry):
    decision = decide_scopes(
        body(system=[{"type": "text", "text": LONG}]),
        prompt_version="assistant@v1",
        registry=registry,
    )
    assert decision.conversation is Scope.TENANT


def test_tool_trust_is_a_deployment_decision(registry: Registry):
    shared = decide_scopes(body(), prompt_version=None, registry=registry)
    assert shared.tools is Scope.SHARED

    # Nothing in the code can detect tool descriptions built from customer data,
    # so the deployment has to say so.
    untrusted = decide_scopes(body(), prompt_version=None, registry=registry, trust_tools=False)
    assert untrusted.tools is Scope.TENANT


def test_the_decision_only_places_a_marker_when_it_can_prove_sharing(
    registry: Registry,
):
    from prism.caching.breakpoints import CachePolicy

    proven, report, decision = apply_to_request(
        body(system=[{"type": "text", "text": LONG}]),
        CachePolicy(),
        prompt_version="assistant@v1",
        registry=registry,
    )
    assert "cache_control" in proven["system"][-1]
    assert report.placements

    unproven, report2, _ = apply_to_request(
        body(system=[{"type": "text", "text": LONG + " extra"}]),
        CachePolicy(),
        prompt_version="assistant@v1",
        registry=registry,
    )
    assert "cache_control" not in unproven["system"][-1]
    assert not report2.placements


# --- token estimation -----------------------------------------------------


def test_the_estimator_scales_with_length():
    short = estimate_text("hello world")
    long = estimate_text("hello world " * 100)
    assert 0 < short < long
    assert long > short * 50


def test_empty_text_costs_nothing():
    assert estimate_text("") == 0


def test_whitespace_runs_do_not_inflate_the_estimate():
    # A run of spaces tokenizes as roughly one token, not one per character.
    assert estimate_text("a" + " " * 200 + "b") < estimate_text("a b " * 50)


def test_non_ascii_costs_more_per_character():
    assert estimate_text("日本語のテキスト") > estimate_text("abcdefgh")


def test_a_request_estimate_covers_tools_system_and_messages():
    minimal = estimate_request(body())
    with_system = estimate_request(body(system=[{"type": "text", "text": LONG}]))
    with_tools = estimate_request(
        body(
            system=[{"type": "text", "text": LONG}],
            tools=[{"name": "f", "description": LONG, "input_schema": {}}],
        )
    )
    assert minimal < with_system < with_tools


def test_an_unknown_block_type_is_estimated_rather_than_skipped():
    # Zero is the one answer guaranteed to be wrong.
    payload = body()
    payload["messages"] = [
        {"role": "user", "content": [{"type": "some_future_block", "data": "x" * 400}]}
    ]
    assert estimate_request(payload) > 50


def test_the_estimator_reports_both_tails_not_a_mean():
    # Breakpoint placement can absorb over-estimation; rate-limit reservation
    # cannot absorb under-estimation. A mean would hide which is happening.
    report = EstimatorReport(
        samples=[EstimatorSample(estimated=90, actual=100)]
        + [EstimatorSample(estimated=105, actual=100) for _ in range(18)]
        + [EstimatorSample(estimated=140, actual=100)]
    )
    p = report.percentiles()
    assert p["max_under"] == pytest.approx(-0.10)
    assert p["max_over"] == pytest.approx(0.40)
    assert p["p50"] == pytest.approx(0.05)


def test_an_estimator_with_too_few_samples_is_not_called_calibrated():
    # Nothing here pretends to accuracy it has not measured.
    assert not EstimatorReport().calibrated
    assert "uncalibrated" in str(EstimatorReport())
    assert EstimatorReport(samples=[EstimatorSample(100, 100) for _ in range(20)]).calibrated


async def test_calibration_compares_against_the_oracle():
    async def oracle(payload):
        return 1234

    report = await calibrate([body(), body()], oracle)
    assert len(report.samples) == 2
    assert all(s.actual == 1234 for s in report.samples)


async def test_a_failing_oracle_call_loses_the_sample_rather_than_the_run():
    calls = {"n": 0}

    async def flaky(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rate limited")
        return 500

    report = await calibrate([body(), body()], flaky)
    assert len(report.samples) == 1
