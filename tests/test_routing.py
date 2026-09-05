"""The router: derived thresholds, two axes, and train/test hygiene."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from prism.routing.economics import EscalationMode, break_even_table, for_tiers
from prism.routing.features import RequestFeatures, extract
from prism.routing.model import DifficultyRouter, RouterUnavailableError, TrainingRow
from prism.routing.policy import RouterPolicy, RoutingDecision, apply_to_body


def body(text: str = "a question", **kwargs):
    out = {
        "model": "claude-opus-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }
    out.update(kwargs)
    return out


# --- the threshold is derived, not tuned ----------------------------------


def test_the_break_even_threshold_is_the_cost_ratio():
    # Route down when p > c/e. Nothing to tune: it falls out of the price list.
    economics = for_tiers("claude-haiku-4-5", "claude-opus-5")
    assert economics.threshold == pytest.approx(economics.cost_ratio)
    # Haiku is ~1/5 the price of Opus, so four wasted Haiku calls still cost
    # less than one avoided Opus call.
    assert economics.threshold == pytest.approx(0.2, abs=0.01)


def test_a_less_dramatic_downgrade_demands_more_confidence():
    haiku = for_tiers("claude-haiku-4-5", "claude-opus-5").threshold
    sonnet = for_tiers("claude-sonnet-5", "claude-opus-5").threshold
    assert sonnet > haiku


def test_expected_cost_beats_the_expensive_tier_exactly_at_the_threshold():
    economics = for_tiers("claude-haiku-4-5", "claude-opus-5")
    at = economics.expected_cost(economics.threshold)
    assert float(at) == pytest.approx(float(economics.expensive_cost), rel=1e-6)
    # Above it, routing down saves; below it, it costs more than going straight up.
    assert economics.saving_vs_expensive(economics.threshold + 0.2) > 0
    assert economics.saving_vs_expensive(economics.threshold - 0.15) < 0


def test_a_verifier_raises_the_bar():
    # A verifier is paid on every request, hit or miss.
    free = for_tiers("claude-haiku-4-5", "claude-opus-5", verifier_fraction=0.0)
    costly = for_tiers("claude-haiku-4-5", "claude-opus-5", verifier_fraction=0.5)
    assert costly.threshold > free.threshold


def test_without_escalation_the_constraint_is_quality_not_money():
    # A bad cheap answer that is simply delivered costs nothing extra, so money
    # always favours routing down and a quality floor has to bind instead.
    economics = for_tiers(
        "claude-haiku-4-5",
        "claude-opus-5",
        mode=EscalationMode.NONE,
        quality_floor=0.9,
    )
    assert economics.threshold == 0.9
    assert economics.expected_cost(0.1) == economics.cheap_cost


def test_the_break_even_table_covers_every_downgrade():
    rows = break_even_table()
    pairs = {(r["cheap"], r["expensive"]) for r in rows}
    assert ("claude-haiku-4-5", "claude-opus-5") in pairs
    assert ("claude-sonnet-5", "claude-opus-5") in pairs


# --- features -------------------------------------------------------------


def test_features_separate_a_lookup_from_a_reasoning_task():
    lookup = extract(body("What is the capital of France?"))
    reasoning = extract(
        body("Prove step by step why this invariant holds, and analyse the tradeoff.")
    )
    names = RequestFeatures.names()
    reasoning_idx = names.index("reasoning_markers")
    lookup_idx = names.index("lookup_markers")

    assert reasoning.metadata[reasoning_idx] > lookup.metadata[reasoning_idx]
    assert lookup.metadata[lookup_idx] > reasoning.metadata[lookup_idx]


def test_tool_complexity_is_a_feature():
    plain = extract(body())
    with_tools = extract(
        body(
            tools=[
                {
                    "name": "f",
                    "input_schema": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                        "required": ["a", "b"],
                    },
                }
            ]
        )
    )
    names = RequestFeatures.names()
    assert with_tools.metadata[names.index("n_tools")] == 1.0
    assert with_tools.metadata[names.index("tool_required_fields")] == 2.0
    assert plain.metadata[names.index("n_tools")] == 0.0


def test_the_embedding_is_projected_down_before_it_reaches_the_model():
    # 384 dimensions against a few hundred rows invites memorisation, and a
    # router that memorises reports a quality-retention figure that is a lie.
    features = RequestFeatures(metadata=[1.0] * 16, embedding=[0.5] * 384)
    assert len(features.vector(project_dimensions=0)) == 16
    assert len(features.vector(project_dimensions=8)) == 24


# --- the classifier -------------------------------------------------------


def synthetic_rows(n: int = 240, seed: int = 0) -> list[TrainingRow]:
    """Short lookups the cheap tier handles; long reasoning tasks it does not."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        easy = i % 2 == 0
        text = (
            f"What is the capital of country number {i}?"
            if easy
            else "Prove step by step why this distributed invariant holds, "
            f"analyse the tradeoff, and design an alternative. Case {i}."
        )
        # 10% label noise, so the model cannot reach a perfect boundary.
        succeeded = easy if rng.random() > 0.10 else not easy
        rows.append(TrainingRow(f"ex-{i}", extract(body(text)), succeeded))
    return rows


def test_the_router_learns_a_separable_signal():
    rows = synthetic_rows()
    train, test = rows[:180], rows[180:]

    router = DifficultyRouter(project_dimensions=0)
    router.fit(train)
    auc, brier, curve = router.evaluate(test)

    assert auc > 0.8
    assert brier < 0.25
    assert curve


def test_an_untrained_router_refuses_to_predict():
    with pytest.raises(RouterUnavailableError, match="not been trained"):
        DifficultyRouter().predict_proba(extract(body()))


def test_training_on_one_outcome_class_is_refused():
    # A boundary cannot be learned from examples that all went the same way, and
    # a model fitted on them would return a constant dressed as a probability.
    rows = [TrainingRow(f"e{i}", extract(body(f"q{i}")), True) for i in range(10)]
    with pytest.raises(ValueError, match="only one outcome class"):
        DifficultyRouter().fit(rows)


def test_coefficients_are_readable():
    # The payoff for picking a linear model: "why was this routed up?" has an
    # answer that fits in a sentence.
    router = DifficultyRouter(project_dimensions=0)
    router.fit(synthetic_rows())
    weights = router.coefficients()
    assert set(weights) == set(RequestFeatures.names())
    assert any(abs(w) > 0.01 for w in weights.values())


def test_a_coin_flip_router_is_flagged_rather_than_shipped():
    from prism.routing.model import RouterReport

    report = RouterReport(n_train=100, n_test=50, auc=0.52, brier=0.25)
    assert not report.beats_base_rate
    assert "coin flip" in report.render()


def test_the_operating_curve_trades_downgrades_against_mistakes():
    router = DifficultyRouter(project_dimensions=0)
    rows = synthetic_rows()
    router.fit(rows[:180])
    _, _, curve = router.evaluate(rows[180:])

    rates = [p.downgrade_rate for p in curve]
    # Raising the threshold can only route fewer requests down.
    assert rates == sorted(rates, reverse=True)


def test_a_saved_router_loads_back(tmp_path):
    router = DifficultyRouter(project_dimensions=0)
    router.fit(synthetic_rows())
    features = extract(body("What is the capital of France?"))
    before = router.predict_proba(features)

    path = tmp_path / "router.joblib"
    router.save(path)
    assert path.with_suffix(".json").is_file()  # readable sidecar

    reloaded = DifficultyRouter.load(path)
    assert reloaded.predict_proba(features) == pytest.approx(before)


# --- the policy -----------------------------------------------------------


class FixedRouter:
    is_trained = True

    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, features):  # noqa: ANN001
        return self.p


def test_a_confident_request_goes_to_the_cheapest_rung():
    policy = RouterPolicy(router=FixedRouter(0.95))
    decision = policy.decide(extract(body()))
    assert decision.model == "claude-haiku-4-5"
    assert "clears" in decision.reason


def test_a_hard_request_goes_to_the_top():
    policy = RouterPolicy(router=FixedRouter(0.05))
    decision = policy.decide(extract(body()))
    assert decision.model == "claude-opus-5"
    assert "clears no downgrade threshold" in decision.reason


def test_the_ladder_is_walked_from_the_bottom():
    # 0.3 clears Haiku-vs-Opus (0.2) but not Sonnet-vs-Opus (0.4), and each rung
    # is its own economic question rather than a bucket of one score.
    policy = RouterPolicy(router=FixedRouter(0.3))
    decision = policy.decide(extract(body()))
    assert decision.model == "claude-haiku-4-5"

    considered = {c["model"]: c for c in decision.considered}
    assert considered["claude-haiku-4-5"]["clears"] is True


def test_effort_varies_within_the_top_tier():
    """Axis 2 has to move independently of axis 1, or it is not an axis.

    Everything reaching the top rung has a low p_success by construction, so
    banding on p_success alone hands out the same effort every time. These two
    both land on Opus and must differ.
    """
    hard = RouterPolicy(router=FixedRouter(0.18)).decide(extract(body()))
    brutal = RouterPolicy(router=FixedRouter(0.01)).decide(extract(body()))
    assert hard.model == brutal.model == "claude-opus-5"
    assert hard.effort == "medium"
    assert brutal.effort == "xhigh"


def test_effort_below_the_top_tier_reflects_the_margin():
    """Sonnet supports effort: barely clearing its break-even buys more thinking.

    The probabilities are derived from the live threshold rather than written in.
    An earlier version hardcoded 0.42 against a threshold of 0.400 -- and then
    Sonnet's introductory pricing expired, the threshold moved to 0.600, and the
    test started failing on a calendar boundary rather than on a code change.
    Deriving it is also the point being tested: the threshold is the cost ratio,
    so it moves when the price list does.
    """
    ladder = ["claude-sonnet-5", "claude-opus-5"]
    threshold = for_tiers("claude-sonnet-5", "claude-opus-5").threshold

    comfortable = RouterPolicy(
        router=FixedRouter(min(0.99, threshold + 0.35)), ladder=list(ladder)
    ).decide(extract(body()))
    marginal = RouterPolicy(router=FixedRouter(threshold + 0.02), ladder=list(ladder)).decide(
        extract(body())
    )

    assert comfortable.model == marginal.model == "claude-sonnet-5"
    assert comfortable.effort == "low"
    assert marginal.effort == "high"


def test_the_threshold_tracks_the_price_list_including_promotions():
    """A derived threshold moves when pricing does; a tuned constant would not.

    Sonnet ran an introductory rate until 2026-08-31. Inside that window the
    Sonnet-to-Opus break-even was 0.400; outside it, 0.600. Any request with a
    predicted success between the two is routed differently depending only on
    the date -- which is correct, and is exactly what a hardcoded constant would
    have got silently wrong.
    """
    from datetime import date

    from prism.registry import MODELS

    sonnet = MODELS["claude-sonnet-5"]
    inside = sonnet.rates(date(2026, 8, 15))
    outside = sonnet.rates(date(2026, 9, 15))
    assert inside == (Decimal("2.00"), Decimal("10.00"))
    assert outside == (Decimal("3.00"), Decimal("15.00"))

    opus_input = MODELS["claude-opus-5"].rates()[0]
    assert float(inside[0] / opus_input) == pytest.approx(0.4, abs=0.01)
    assert float(outside[0] / opus_input) == pytest.approx(0.6, abs=0.01)


def test_effort_is_omitted_on_tiers_that_do_not_support_it():
    decision = RouterPolicy(router=FixedRouter(0.99)).decide(extract(body()))
    assert decision.model == "claude-haiku-4-5"
    assert decision.effort is None  # Haiku takes a token budget instead


def test_an_untrained_router_defaults_upward_not_downward():
    # Falling back to the cheap tier would silently degrade quality the moment
    # the model file went missing.
    decision = RouterPolicy(router=None).decide(extract(body()))
    assert decision.model == "claude-opus-5"
    assert "rather than silently degrading quality" in decision.reason


def test_escalation_goes_to_the_top_not_the_next_rung():
    policy = RouterPolicy(router=FixedRouter(0.9))
    first = policy.decide(extract(body()))
    assert first.model == "claude-haiku-4-5"

    # Having already paid for one failure, a second costs more than the gap
    # between the middle and top tiers.
    escalated = policy.escalate(first)
    assert escalated.model == "claude-opus-5"
    assert escalated.escalated_from == "claude-haiku-4-5"


def test_the_decision_rewrites_the_body_for_the_chosen_tier():
    decision = RoutingDecision(
        model="claude-haiku-4-5", effort="low", p_success=0.9, threshold=0.2, reason=""
    )
    rewritten = apply_to_body(body(max_tokens=100_000, temperature=0.7), decision)

    assert rewritten["model"] == "claude-haiku-4-5"
    # Clamped to the tier's ceiling: a request sized for Opus would 400 on Haiku
    # and the failure would look like a routing bug.
    assert rewritten["max_tokens"] == 64_000
    # Haiku accepts sampling params, so temperature survives.
    assert rewritten["temperature"] == 0.7


def test_routing_up_strips_params_the_target_tier_rejects():
    decision = RoutingDecision(
        model="claude-opus-5", effort="high", p_success=0.1, threshold=0.2, reason=""
    )
    rewritten = apply_to_body(body(temperature=0.7), decision)
    assert "temperature" not in rewritten
    assert rewritten["output_config"]["effort"] == "high"


# --- through the gateway --------------------------------------------------


@pytest.fixture
def routed_client(provider, recorder, settings, semantic, chain):
    """A client whose router always predicts an easy request."""
    from fastapi.testclient import TestClient

    from prism.api.deps import (
        get_chain,
        get_provider,
        get_recorder,
        get_router_policy,
        get_semantic_cache,
    )
    from prism.main import create_app

    app = create_app(settings)
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_recorder] = lambda: recorder
    app.dependency_overrides[get_semantic_cache] = lambda: semantic
    app.dependency_overrides[get_chain] = lambda: chain
    app.dependency_overrides[get_router_policy] = lambda: RouterPolicy(router=FixedRouter(0.95))
    with TestClient(app) as c:
        yield c


def test_prism_auto_hands_tier_selection_to_the_router(routed_client, provider, recorder):
    resp = routed_client.post(
        "/v1/chat/completions",
        json={
            "model": "prism-auto",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
    )
    assert resp.status_code == 200
    # The alias resolves to Opus, but the router moved it down.
    assert provider.last_call["model"] == "claude-haiku-4-5"
    assert resp.headers["x-prism-model"] == "claude-haiku-4-5"
    assert recorder.last.extra["routing"]["model"] == "claude-haiku-4-5"


def test_a_pinned_model_is_never_rerouted(routed_client, provider, recorder):
    # Routing is opt-in per request: a client that named a tier keeps it, even
    # with a router loaded that would have chosen otherwise.
    routed_client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
    )
    assert provider.last_call["model"] == "claude-opus-5"
    assert "routing" not in recorder.last.extra


def test_the_routing_decision_and_its_alternatives_land_on_the_trace(routed_client, recorder):
    routed_client.post(
        "/v1/chat/completions",
        json={
            "model": "prism-auto",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
    )
    routing = recorder.last.extra["routing"]
    assert routing["p_success"] == pytest.approx(0.95)
    # What was considered, not just what was chosen — otherwise a surprising
    # route cannot be explained after the fact.
    assert routing["considered"]
    assert "expected_cost_usd" in routing["considered"][0]


def test_rerouting_retranslates_rather_than_rewriting(routed_client, provider):
    # Haiku accepts temperature; Opus rejects it. Translating for Opus and then
    # rewriting the model field would drop it on a request that could keep it.
    routed_client.post(
        "/v1/chat/completions",
        json={
            "model": "prism-auto",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "temperature": 0.7,
        },
    )
    sent = provider.last_call
    assert sent["model"] == "claude-haiku-4-5"
    assert sent["temperature"] == 0.7
