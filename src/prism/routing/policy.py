"""The routing decision: which tier, and how much thinking.

Most routers pick a model. This one picks a model *and* a reasoning budget,
which matters because the two axes have different shapes:

* **Tier is discrete.** Three rungs, and each downgrade has its own break-even
  probability from `economics.py`.
* **Effort is continuous.** Thinking tokens bill as output, so effort is a dial
  rather than a jump — and because it multiplies the *output* price rather than
  switching price lists, lowering effort on an expensive tier can be cheaper than
  moving to a cheap tier at full effort. The router should be able to prefer
  that, so effort is chosen after the tier rather than bolted on.

The decision walks the ladder from the bottom: take the cheapest tier whose
predicted success clears its own break-even threshold. That is not the same as
"predict a difficulty score and bucket it" — each rung is a separate economic
question, and the thresholds are not evenly spaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..registry import MODELS, ModelSpec
from .economics import EscalationMode, RoutingEconomics, for_tiers
from .features import RequestFeatures
from .model import DifficultyRouter

Effort = Literal["low", "medium", "high", "xhigh", "max"]

#: Bottom-up. The router takes the first rung that clears its threshold.
LADDER = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]

#: Effort means different things at different rungs, so the bands differ.
#:
#: Banding on `p_success` alone does not work: `p_success` is also what chose
#: the tier, so within a tier its range is already constrained. Everything that
#: reaches the top rung has a low `p` by construction, and a single band table
#: would hand out "high" every time — an axis that never moves.
#:
#: At the top, the request is hard by definition and the question is *how* hard.
#: Below the top, the tier's break-even was cleared, so the question is by what
#: margin — a request that only just qualified gets more thinking than one that
#: sailed past.
_TOP_TIER_BANDS: list[tuple[float, Effort]] = [
    (0.15, "medium"),
    (0.05, "high"),
    (0.00, "xhigh"),
]
_MARGIN_BANDS: list[tuple[float, Effort]] = [
    (0.20, "low"),
    (0.05, "medium"),
    (-1.0, "high"),
]


@dataclass
class RoutingDecision:
    model: str
    effort: Effort | None
    p_success: float
    threshold: float
    reason: str
    considered: list[dict[str, Any]] = field(default_factory=list)
    escalated_from: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "p_success": self.p_success,
            "threshold": self.threshold,
            "reason": self.reason,
            "considered": self.considered,
            "escalated_from": self.escalated_from,
        }


@dataclass
class RouterPolicy:
    router: DifficultyRouter | None = None
    ladder: list[str] = field(default_factory=lambda: list(LADDER))
    mode: EscalationMode = EscalationMode.VERIFY_THEN_ESCALATE
    verifier_fraction: float = 0.0
    quality_floor: float = 0.85
    #: Where an untrained or unavailable router lands. The top of the ladder:
    #: falling back to the cheap tier would silently degrade quality the moment
    #: the model file went missing.
    fallback: str = "claude-opus-5"
    #: Effort applies only to tiers that support it; Haiku takes a token budget
    #: instead, which the translation layer handles.
    choose_effort: bool = True

    def economics_for(self, cheap: str) -> RoutingEconomics:
        return for_tiers(
            cheap,
            self.ladder[-1],
            mode=self.mode,
            verifier_fraction=self.verifier_fraction,
            quality_floor=self.quality_floor,
        )

    def decide(self, features: RequestFeatures) -> RoutingDecision:
        if self.router is None or not self.router.is_trained:
            top = MODELS[self.fallback]
            return RoutingDecision(
                model=top.id,
                effort="high" if top.supports_effort and self.choose_effort else None,
                p_success=0.0,
                threshold=0.0,
                reason="no trained router; defaulting to the top of the ladder "
                "rather than silently degrading quality",
            )

        p_success = self.router.predict_proba(features)
        considered: list[dict[str, Any]] = []

        for tier in self.ladder[:-1]:
            economics = self.economics_for(tier)
            clears = economics.worth_routing_down(p_success)
            considered.append(
                {
                    "model": tier,
                    "threshold": economics.threshold,
                    "clears": clears,
                    "expected_cost_usd": str(economics.expected_cost(p_success)),
                }
            )
            if clears:
                return RoutingDecision(
                    model=tier,
                    effort=self._effort_for(MODELS[tier], p_success, threshold=economics.threshold),
                    p_success=p_success,
                    threshold=economics.threshold,
                    reason=(
                        f"P(success)={p_success:.3f} clears the {tier} break-even "
                        f"of {economics.threshold:.3f} (cost ratio against "
                        f"{self.ladder[-1]})"
                    ),
                    considered=considered,
                )

        top = MODELS[self.ladder[-1]]
        return RoutingDecision(
            model=top.id,
            effort=self._effort_for(top, p_success, is_top=True),
            p_success=p_success,
            threshold=1.0,
            reason=(
                f"P(success)={p_success:.3f} clears no downgrade threshold; "
                "the expected cost of a retry exceeds going straight to the top"
            ),
            considered=considered,
        )

    def _effort_for(
        self,
        spec: ModelSpec,
        p_success: float,
        *,
        threshold: float | None = None,
        is_top: bool = False,
    ) -> Effort | None:
        """Axis 2: how much thinking, given where on the ladder we landed.

        Thinking tokens bill as output, so this is a continuous dial rather than
        a discrete jump — and because it scales the output price instead of
        switching price lists, trimming effort on an expensive tier can beat
        moving to a cheap tier at full effort.
        """
        if not self.choose_effort or not spec.supports_effort:
            return None
        if is_top or threshold is None:
            for floor, effort in _TOP_TIER_BANDS:
                if p_success >= floor:
                    return effort
            return "xhigh"
        margin = p_success - threshold
        for floor, effort in _MARGIN_BANDS:
            if margin >= floor:
                return effort
        return "high"

    def escalate(self, decision: RoutingDecision) -> RoutingDecision:
        """Move one rung up after a verifier rejected the cheap answer.

        Escalation goes to the *top*, not the next rung. Having already paid for
        one failure, a second failure costs more than the difference between the
        middle and top tiers — the expected-cost arithmetic that justified
        routing down no longer applies once it has been shown wrong.
        """
        top = MODELS[self.ladder[-1]]
        return RoutingDecision(
            model=top.id,
            effort="high" if top.supports_effort and self.choose_effort else None,
            p_success=decision.p_success,
            threshold=decision.threshold,
            reason="verifier rejected the cheap answer; escalating to the top",
            considered=decision.considered,
            escalated_from=decision.model,
        )


def apply_to_body(body: dict[str, Any], decision: RoutingDecision) -> dict[str, Any]:
    """Rewrite an upstream body to match the routing decision."""
    out = dict(body)
    out["model"] = decision.model
    spec = MODELS[decision.model]

    if decision.effort and spec.supports_effort:
        config = dict(out.get("output_config") or {})
        config["effort"] = decision.effort
        out["output_config"] = config

    # Clamp to the chosen tier's ceiling: a request sized for Opus can exceed
    # what Haiku will accept, and a 400 here would look like a routing bug.
    if out.get("max_tokens", 0) > spec.max_output_tokens:
        out["max_tokens"] = spec.max_output_tokens
    if not spec.supports_sampling_params:
        for key in ("temperature", "top_p"):
            out.pop(key, None)
    return out
