"""Position bias, verdict parsing, and judge calibration."""

from __future__ import annotations

import json

import pytest

from prism.eval.calibration import (
    CalibrationReport,
    HumanLabel,
    agreement_interval,
    calibrate,
)
from prism.eval.dataset import Example
from prism.eval.judge import (
    PairwiseJudge,
    PairwiseResult,
    Verdict,
    parse_verdict,
    position_bias_rate,
    render_request,
    summarize,
)


def example(eid: str = "ex1") -> Example:
    return Example.model_validate(
        {
            "id": eid,
            "split": "test",
            "task_type": "open_ended",
            "messages": [{"role": "user", "content": "Explain gradient descent."}],
        }
    )


class ScriptedJudge:
    """Returns a queued verdict per call, so position bias can be simulated."""

    model = "fake-judge-v1"

    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = list(verdicts)
        self.prompts: list[str] = []

    async def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        winner = self.verdicts.pop(0)
        return json.dumps({"winner": winner, "reason": "because"})


class AlwaysFirstJudge:
    """The pathological case: picks whatever it reads first, every time."""

    model = "position-biased"

    async def complete(self, system: str, user: str) -> str:
        return json.dumps({"winner": "A", "reason": "first one looked good"})


class BrokenJudge:
    model = "broken"

    async def complete(self, system: str, user: str) -> str:
        raise RuntimeError("upstream down")


# --- position bias --------------------------------------------------------


async def test_every_comparison_runs_in_both_orders():
    judge = ScriptedJudge(["A", "B"])
    await PairwiseJudge(judge).compare(example(), "candidate text", "baseline text")

    assert len(judge.prompts) == 2
    # Same two responses, swapped between the A and B slots.
    first, second = judge.prompts
    assert first.index("candidate text") < first.index("baseline text")
    assert second.index("baseline text") < second.index("candidate text")


async def test_a_consistent_win_survives_the_swap():
    # Candidate is A in order one and B in order two; picking it both times is
    # the only thing that counts as a win.
    result = await PairwiseJudge(ScriptedJudge(["A", "B"])).compare(
        example(), "candidate", "baseline"
    )
    assert result.verdict is Verdict.A
    assert result.candidate_score == 1.0


async def test_a_consistent_loss_survives_the_swap():
    result = await PairwiseJudge(ScriptedJudge(["B", "A"])).compare(
        example(), "candidate", "baseline"
    )
    assert result.verdict is Verdict.B
    assert result.candidate_score == 0.0


async def test_a_judge_that_always_picks_the_first_scores_nothing():
    # The whole point of swapping. Without it this judge would report a 100%
    # win rate for whichever arm happened to be printed first.
    result = await PairwiseJudge(AlwaysFirstJudge()).compare(example(), "candidate", "baseline")
    assert result.verdict is Verdict.INCONSISTENT
    # A judge that cannot hold an opinion has not detected a difference.
    assert result.candidate_score == 0.5


async def test_double_tie_is_a_tie():
    result = await PairwiseJudge(ScriptedJudge(["tie", "tie"])).compare(
        example(), "candidate", "baseline"
    )
    assert result.verdict is Verdict.TIE
    assert result.candidate_score == 0.5


async def test_a_tie_in_one_order_only_is_inconsistent_not_a_win():
    # Half an opinion is not an opinion; counting it as a win would manufacture
    # signal out of the judge's own instability.
    result = await PairwiseJudge(ScriptedJudge(["A", "tie"])).compare(
        example(), "candidate", "baseline"
    )
    assert result.verdict is Verdict.INCONSISTENT


async def test_a_failed_judge_call_is_an_error_not_a_tie():
    # Silently scoring 0.5 would let an outage look like "no difference found".
    result = await PairwiseJudge(BrokenJudge()).compare(example(), "cand", "base")
    assert result.verdict is Verdict.ERROR
    assert not result.is_usable


async def test_unparseable_output_is_an_error():
    class Rambling:
        model = "rambling"

        async def complete(self, system: str, user: str) -> str:
            return "Well, it depends on what you mean by better..."

    result = await PairwiseJudge(Rambling()).compare(example(), "cand", "base")
    assert result.verdict is Verdict.ERROR


def test_position_bias_rate_is_reported_on_its_own():
    results = [
        PairwiseResult("a", Verdict.A),
        PairwiseResult("b", Verdict.INCONSISTENT),
        PairwiseResult("c", Verdict.INCONSISTENT),
        PairwiseResult("d", Verdict.TIE),
        PairwiseResult("e", Verdict.ERROR),  # excluded from the denominator
    ]
    assert position_bias_rate(results) == pytest.approx(0.5)

    summary = summarize(results)
    assert summary == {
        "comparisons": 5,
        "usable": 4,
        "errors": 1,
        "candidate_wins": 1,
        "baseline_wins": 0,
        "ties": 1,
        "inconsistent": 2,
        "position_bias_rate": pytest.approx(0.5),
    }


# --- parsing --------------------------------------------------------------


def test_verdict_parsing_tolerates_surrounding_prose():
    winner, reason = parse_verdict('Sure!\n{"winner": "B", "reason": "clearer"}\nDone.')
    assert winner == "B"
    assert reason == "clearer"


def test_an_invalid_winner_token_is_rejected():
    assert parse_verdict('{"winner": "maybe", "reason": "x"}')[0] is None
    assert parse_verdict("not json at all")[0] is None


def test_the_request_shown_to_the_judge_includes_the_conversation():
    rendered = render_request(example())
    assert "user: Explain gradient descent." in rendered


# --- calibration ----------------------------------------------------------


def test_kappa_is_computed_against_human_labels_on_shared_examples():
    judge_results = [
        PairwiseResult("e1", Verdict.A),
        PairwiseResult("e2", Verdict.B),
        PairwiseResult("e3", Verdict.TIE),
        PairwiseResult("e4", Verdict.A),
    ]
    humans = [
        HumanLabel("e1", "A"),
        HumanLabel("e2", "B"),
        HumanLabel("e3", "tie"),
        HumanLabel("e4", "A"),
    ]
    report = calibrate(judge_results, humans, judge_model="fake-judge-v1")
    assert report.n == 4
    assert report.kappa == pytest.approx(1.0)
    assert report.raw_agreement == 1.0
    assert report.trustworthy
    assert report.judge_model == "fake-judge-v1"


def test_inconsistent_verdicts_are_scored_as_ties_against_humans():
    # The human never saw two orderings, so they have no "inconsistent"
    # category. Scoring against a label they could not produce would depress
    # kappa for the wrong reason; the bias rate is reported separately.
    judge_results = [PairwiseResult("e1", Verdict.INCONSISTENT)]
    report = calibrate(judge_results, [HumanLabel("e1", "tie")])
    assert report.raw_agreement == 1.0
    assert report.position_bias_rate == 1.0


def test_a_judge_that_agrees_by_luck_gets_a_low_kappa():
    # 8 of 10 agree, but both sides say "tie" most of the time.
    judge_results = [PairwiseResult(f"e{i}", Verdict.TIE) for i in range(8)] + [
        PairwiseResult("e8", Verdict.A),
        PairwiseResult("e9", Verdict.B),
    ]
    humans = [HumanLabel(f"e{i}", "tie") for i in range(8)] + [
        HumanLabel("e8", "B"),
        HumanLabel("e9", "A"),
    ]
    report = calibrate(judge_results, humans)
    assert report.raw_agreement == pytest.approx(0.8)
    assert not report.trustworthy  # 80% agreement, but kappa says otherwise


def test_calibration_requires_overlap():
    with pytest.raises(ValueError, match="no overlap"):
        calibrate([PairwiseResult("e1", Verdict.A)], [HumanLabel("other", "A")])


def test_errored_comparisons_are_excluded_from_calibration():
    judge_results = [
        PairwiseResult("e1", Verdict.A),
        PairwiseResult("e2", Verdict.ERROR),
    ]
    report = calibrate(judge_results, [HumanLabel("e1", "A"), HumanLabel("e2", "B")])
    assert report.n == 1


def test_agreement_at_the_usual_label_budget_carries_uncertainty():
    report = CalibrationReport(
        n=100,
        kappa=0.62,
        raw_agreement=0.78,
        interpretation="substantial",
        position_bias_rate=0.08,
        confusion={},
    )
    ci = agreement_interval(report)
    # ~100 labels buys roughly +-10 points; reporting 78% bare overstates it.
    assert ci.low < 0.72
    assert ci.high > 0.84


def test_a_human_label_outside_the_allowed_set_is_rejected():
    with pytest.raises(ValueError, match="label must be one of"):
        HumanLabel("e1", "inconsistent")
