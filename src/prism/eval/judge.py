"""Pairwise LLM-as-judge.

Three biases make a naive judge worse than useless, and each one is handled
structurally here rather than hoped away.

**Position bias.** LLM judges systematically favour whichever answer they read
first. So every comparison runs twice with the responses swapped, and the pair
is only a win if the judge picks the *same response* in both orders. Disagree
across orders and it is recorded as ``inconsistent`` — which is information, not
noise: a rising inconsistency rate means the judge cannot tell the arms apart,
and the honest reading of that is "no measurable difference", not a coin flip.

**Self-preference.** Models score their own output higher. Prism serves Claude
across every tier, so the judge must come from a different family. Staying
single-vendor on the serving side makes this structural rather than a discipline
you have to remember — there is no configuration in which the judge and the
system under test are the same model.

**Verbosity bias.** Judges reward length. The rubric names it explicitly and the
judge is asked for a verdict token rather than an essay, which also makes
parsing deterministic.

None of that makes the judge *correct*. That is what ``calibration.py`` and
Cohen's kappa against human labels are for.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .dataset import Example


class Verdict(StrEnum):
    A = "A"
    B = "B"
    TIE = "tie"
    INCONSISTENT = "inconsistent"
    ERROR = "error"


JUDGE_SYSTEM_PROMPT = """\
You are evaluating two candidate responses to the same request.

Judge only on: correctness, faithfulness to the request, and usefulness to the \
person who asked. Explicitly ignore length, formatting flourish, and confident \
tone — a shorter response that is correct beats a longer one that is padded.

Reply with a single JSON object and nothing else:
{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}

Use "tie" when the two are of genuinely equal quality, including when both are \
equally wrong."""

_JUDGE_TEMPLATE = """\
<request>
{request}
</request>

<response_A>
{a}
</response_A>

<response_B>
{b}
</response_B>"""


class JudgeClient(Protocol):
    """Minimal completion interface, so the judge can be faked in tests.

    Deliberately not the gateway's own provider type: the judge is a *client* of
    a model, not a proxy for one, and wiring it through Prism's egress would
    make eval runs depend on the system under test.
    """

    model: str

    async def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class PairwiseResult:
    example_id: str
    verdict: Verdict
    # Raw per-order calls, kept so an inconsistent verdict can be inspected
    # rather than merely counted.
    first_order: str | None = None
    second_order: str | None = None
    reason: str = ""

    @property
    def candidate_score(self) -> float:
        """Win = 1, tie = 0.5, loss = 0 — the standard pairwise scoring.

        Inconsistent counts as a tie for the aggregate. A judge that cannot hold
        an opinion when the order flips has not detected a difference, and
        scoring it as anything else would manufacture signal.
        """
        if self.verdict is Verdict.A:
            return 1.0
        if self.verdict is Verdict.B:
            return 0.0
        return 0.5

    @property
    def is_usable(self) -> bool:
        return self.verdict is not Verdict.ERROR


def render_request(example: Example) -> str:
    """Flatten an example's messages into the prompt the judge sees."""
    parts: list[str] = []
    for message in example.messages:
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        parts.append(f"{message.get('role', 'user')}: {content}")
    return "\n".join(parts)


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_verdict(raw: str) -> tuple[str | None, str]:
    """Extract (winner, reason). Returns (None, ...) when unparseable."""
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return None, ""
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, ""
    winner = payload.get("winner")
    if winner not in ("A", "B", "tie"):
        return None, str(payload.get("reason", ""))
    return winner, str(payload.get("reason", ""))


class PairwiseJudge:
    def __init__(self, client: JudgeClient) -> None:
        self.client = client

    async def compare(self, example: Example, candidate: str, baseline: str) -> PairwiseResult:
        """Score one candidate against one baseline, in both orders."""
        request = render_request(example)

        # Order 1: candidate is A. Order 2: candidate is B. A judge with no
        # position bias gives mirrored answers; anything else is bias, not signal.
        forward = await self._ask(request, candidate, baseline)
        reverse = await self._ask(request, baseline, candidate)

        if forward is None or reverse is None:
            return PairwiseResult(example.id, Verdict.ERROR, forward, reverse)

        forward_winner, forward_reason = parse_verdict(forward)
        reverse_winner, reverse_reason = parse_verdict(reverse)
        if forward_winner is None or reverse_winner is None:
            return PairwiseResult(example.id, Verdict.ERROR, forward, reverse)

        # Translate both into "did the candidate win?", undoing the swap.
        candidate_won_forward = forward_winner == "A"
        candidate_won_reverse = reverse_winner == "B"
        tie_forward = forward_winner == "tie"
        tie_reverse = reverse_winner == "tie"

        if tie_forward and tie_reverse:
            verdict = Verdict.TIE
        elif candidate_won_forward and candidate_won_reverse:
            verdict = Verdict.A
        elif (
            not candidate_won_forward
            and not candidate_won_reverse
            and not (tie_forward or tie_reverse)
        ):
            verdict = Verdict.B
        else:
            verdict = Verdict.INCONSISTENT

        return PairwiseResult(
            example_id=example.id,
            verdict=verdict,
            first_order=forward,
            second_order=reverse,
            reason=forward_reason or reverse_reason,
        )

    async def _ask(self, request: str, a: str, b: str) -> str | None:
        try:
            return await self.client.complete(
                JUDGE_SYSTEM_PROMPT,
                _JUDGE_TEMPLATE.format(request=request, a=a, b=b),
            )
        except Exception:  # noqa: BLE001 — one bad call must not sink the run
            return None


def position_bias_rate(results: Sequence[PairwiseResult]) -> float:
    """Share of comparisons where the judge flipped when the order flipped.

    Worth reporting on its own. A judge with a 30% inconsistency rate is not
    measuring quality, and no amount of downstream statistics repairs that.
    """
    usable = [r for r in results if r.is_usable]
    if not usable:
        return float("nan")
    return sum(1 for r in usable if r.verdict is Verdict.INCONSISTENT) / len(usable)


def summarize(results: Sequence[PairwiseResult]) -> dict[str, Any]:
    usable = [r for r in results if r.is_usable]
    return {
        "comparisons": len(results),
        "usable": len(usable),
        "errors": len(results) - len(usable),
        "candidate_wins": sum(1 for r in usable if r.verdict is Verdict.A),
        "baseline_wins": sum(1 for r in usable if r.verdict is Verdict.B),
        "ties": sum(1 for r in usable if r.verdict is Verdict.TIE),
        "inconsistent": sum(1 for r in usable if r.verdict is Verdict.INCONSISTENT),
        "position_bias_rate": position_bias_rate(results),
    }
