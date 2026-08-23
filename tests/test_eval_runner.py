"""Dataset validation, metrics, and end-to-end run orchestration."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from prism.eval import metrics
from prism.eval.dataset import Dataset, DatasetError, Example, assert_disjoint, load
from prism.eval.judge import PairwiseJudge
from prism.eval.runner import Arm, Response, run

GOLDEN = Path(__file__).resolve().parents[1] / "datasets" / "golden" / "v1.jsonl"


# --- dataset --------------------------------------------------------------


def test_the_shipped_golden_set_is_valid():
    dataset = load(GOLDEN)
    assert len(dataset) > 0
    assert dataset.judged().examples
    assert dataset.programmatic().examples


def test_examples_carry_their_own_scoring_method():
    # Scoring lives in the data so a run cannot quietly promote hard examples to
    # the judge or demote open-ended ones to exact match.
    dataset = load(GOLDEN)
    assert all(e.is_judged for e in dataset.judged())
    assert all(not e.is_judged for e in dataset.programmatic())


def test_a_non_open_ended_example_needs_an_answer_key():
    with pytest.raises(ValueError, match="need references or a schema"):
        Example.model_validate(
            {
                "id": "x",
                "split": "test",
                "task_type": "qa",
                "messages": [{"role": "user", "content": "?"}],
            }
        )


def test_schema_conformance_requires_a_schema():
    with pytest.raises(ValueError, match="requires a schema"):
        Example.model_validate(
            {
                "id": "x",
                "split": "test",
                "task_type": "extraction",
                "messages": [{"role": "user", "content": "?"}],
                "references": ["{}"],
                "metrics": ["schema_conformance"],
            }
        )


def test_duplicate_ids_across_splits_are_rejected(tmp_path):
    # The same prompt in two splits is exactly the contamination this catches.
    path = tmp_path / "dup.jsonl"
    row = {
        "id": "same",
        "task_type": "qa",
        "messages": [{"role": "user", "content": "?"}],
        "references": ["a"],
        "metrics": ["exact_match"],
    }
    path.write_text(
        json.dumps({**row, "split": "test"})
        + "\n"
        + json.dumps({**row, "split": "router_train"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate id"):
        load(path)


def test_the_router_train_test_boundary_is_guarded():
    # The router trains on eval outcomes. Training and scoring on the same
    # examples turns quality retention into memorisation.
    dataset = load(GOLDEN)
    assert_disjoint(dataset.split("router_train"), dataset.split("test"))

    shared = Dataset(list(dataset.split("test")))
    with pytest.raises(DatasetError, match="share"):
        assert_disjoint(shared, dataset.split("test"))


# --- metrics --------------------------------------------------------------


def test_exact_match_ignores_formatting_habits():
    # Without normalisation, exact match scores prose style rather than truth.
    assert metrics.exact_match("The answer is Paris.", "paris") == 0.0
    assert metrics.exact_match("  the Paris ", "Paris") == 1.0
    assert (
        metrics.any_exact_match("Mary Shelley", ["Mary Wollstonecraft Shelley", "Mary Shelley"])
        == 1.0
    )


def test_token_f1_gives_partial_credit():
    assert metrics.token_f1("Mary Shelley", "Mary Wollstonecraft Shelley") == pytest.approx(0.8)
    assert metrics.token_f1("completely different", "Mary Shelley") == 0.0


def test_schema_conformance_checks_types_required_and_enums():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "plan": {"type": "string", "enum": ["Free", "Pro"]},
            "seats": {"type": "integer"},
        },
        "required": ["name", "plan"],
        "additionalProperties": False,
    }
    assert metrics.schema_conformance('{"name": "Jo", "plan": "Pro"}', schema) == 1.0
    assert metrics.schema_conformance('{"name": "Jo"}', schema) == 0.0  # missing required
    assert metrics.schema_conformance('{"name": "Jo", "plan": "Gold"}', schema) == 0.0  # enum
    assert metrics.schema_conformance('{"name": "Jo", "plan": "Pro", "x": 1}', schema) == 0.0
    assert metrics.schema_conformance("not json", schema) == 0.0


def test_booleans_are_not_integers_for_schema_purposes():
    # bool subclasses int in Python; JSON Schema does not agree.
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert metrics.schema_conformance('{"n": true}', schema) == 0.0
    assert metrics.schema_conformance('{"n": 3}', schema) == 1.0


# --- runner ---------------------------------------------------------------


def tiny_dataset() -> Dataset:
    return Dataset(
        [
            Example.model_validate(
                {
                    "id": f"q{i}",
                    "split": "test",
                    "task_type": "qa",
                    "messages": [{"role": "user", "content": f"q{i}"}],
                    "references": ["right"],
                    "metrics": ["exact_match"],
                }
            )
            for i in range(20)
        ]
    )


def arm_answering(correct_upto: int, name: str, *, cost: str = "0.001") -> Arm:
    async def fn(example: Example) -> Response:
        index = int(example.id[1:])
        return Response(
            example_id=example.id,
            text="right" if index < correct_upto else "wrong",
            model="fake",
            cost_usd=Decimal(cost),
        )

    return Arm(name=name, run=fn)


async def test_a_single_arm_run_reports_an_interval_not_a_point():
    report = await run(tiny_dataset(), arm_answering(14, "candidate"))
    interval = report.metric_intervals["exact_match"]
    assert interval.point == pytest.approx(0.7)
    assert interval.low < interval.point < interval.high
    assert not report.regressed  # no baseline, nothing to regress against


async def test_a_paired_run_reports_a_delta_and_fails_the_gate_on_a_real_drop():
    report = await run(
        tiny_dataset(),
        arm_answering(4, "candidate"),
        baseline=arm_answering(18, "baseline"),
    )
    delta = report.metric_deltas["exact_match"]
    assert delta.point < 0
    assert report.regressed
    assert "FAIL" in report.render()


async def test_an_identical_arm_does_not_trip_the_gate():
    report = await run(
        tiny_dataset(),
        arm_answering(14, "candidate"),
        baseline=arm_answering(14, "baseline"),
    )
    assert report.metric_deltas["exact_match"].point == 0.0
    assert not report.regressed
    assert "pass" in report.render()


async def test_tolerance_permits_a_declared_quality_for_cost_tradeoff():
    strict = await run(
        tiny_dataset(),
        arm_answering(12, "cheap"),
        baseline=arm_answering(16, "frontier"),
    )
    assert strict.regressed

    lenient = await run(
        tiny_dataset(),
        arm_answering(12, "cheap"),
        baseline=arm_answering(16, "frontier"),
        tolerance=0.30,
    )
    assert not lenient.regressed


async def test_a_failing_arm_scores_zero_rather_than_being_dropped():
    # Dropping errors would let an arm improve its average by failing on the
    # examples it finds hard.
    async def flaky(example: Example) -> Response:
        if int(example.id[1:]) % 2 == 0:
            raise RuntimeError("boom")
        return Response(example.id, "right")

    report = await run(tiny_dataset(), Arm("flaky", flaky))
    assert report.candidate.failures == 10
    assert report.metric_intervals["exact_match"].point == pytest.approx(0.5)


async def test_cost_per_successful_task_penalises_a_cheap_failing_arm():
    # A cheap arm that fails most of the time is not cheap.
    cheap = await run(tiny_dataset(), arm_answering(4, "cheap", cost="0.001"))
    dear = await run(tiny_dataset(), arm_answering(20, "frontier", cost="0.004"))

    assert cheap.candidate.total_cost_usd < dear.candidate.total_cost_usd
    # ...but per successful task the ordering flips.
    assert cheap.candidate.cost_per_successful_task > dear.candidate.cost_per_successful_task


async def test_judged_examples_only_run_when_a_baseline_exists():
    class Judge:
        model = "fake"

        async def complete(self, system: str, user: str) -> str:
            return json.dumps({"winner": "A", "reason": "x"})

    dataset = Dataset(
        [
            Example.model_validate(
                {
                    "id": "o1",
                    "split": "test",
                    "task_type": "open_ended",
                    "messages": [{"role": "user", "content": "explain"}],
                }
            )
        ]
    )

    async def answer(example: Example) -> Response:
        return Response(example.id, "an answer")

    solo = await run(dataset, Arm("candidate", answer), judge=PairwiseJudge(Judge()))
    assert solo.judge_results == []  # a pairwise comparison needs two arms

    paired = await run(
        dataset,
        Arm("candidate", answer),
        baseline=Arm("baseline", answer),
        judge=PairwiseJudge(Judge()),
    )
    # This judge always picks whatever it reads first, so swapping catches it.
    assert paired.judge_summary["inconsistent"] == 1
    assert paired.judge_summary["position_bias_rate"] == 1.0


async def test_the_gate_report_is_ascii_only():
    """CI logs and Windows consoles are cp1252.

    A stray glyph in the report turns a passing gate into a UnicodeEncodeError,
    which is a build failure that says nothing about quality. Found the hard way:
    the delta and kappa symbols used to crash `python scripts/eval.py` on Windows.
    """
    report = await run(
        tiny_dataset(),
        arm_answering(4, "candidate"),
        baseline=arm_answering(18, "baseline"),
    )
    rendered = report.render()
    assert rendered.isascii(), [c for c in rendered if not c.isascii()]
    rendered.encode("cp1252")  # raises if a glyph sneaks back in


def test_calibration_summary_is_ascii_only():
    from prism.eval.calibration import CalibrationReport

    text = str(
        CalibrationReport(
            n=100,
            kappa=0.62,
            raw_agreement=0.78,
            interpretation="substantial",
            position_bias_rate=0.08,
            confusion={},
        )
    )
    assert text.isascii()
    text.encode("cp1252")
