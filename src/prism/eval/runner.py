"""Orchestrates one evaluation run.

An arm is anything that can answer an example — a prompt version, a model tier,
a routing policy, a cache configuration. The runner does not care which; it
scores whatever it is handed. That is deliberate, because the same harness has
to serve four later phases: the CI gate (week 6) compares prompt versions, the
cache work (weeks 7-8) compares cache configurations, the router (week 9)
compares tiers, and none of them should need their own scoring code.

Programmatic metrics are computed for every example that has an answer key.
Judged comparisons run only on the open-ended residual, and only when a baseline
arm exists — there is no such thing as a pairwise comparison with one arm.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from .calibration import CalibrationReport
from .dataset import Dataset, Example
from .judge import PairwiseJudge, PairwiseResult, summarize
from .metrics import SCORERS
from .stats import Interval, bootstrap_ci, paired_bootstrap_ci, regression_detected


@dataclass
class Response:
    """One arm's answer to one example."""

    example_id: str
    text: str
    model: str = ""
    cost_usd: Decimal = Decimal(0)
    latency_ms: int = 0
    trace_id: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# An arm is a name plus a way to answer an example.
ArmFn = Callable[[Example], Awaitable[Response]]


@dataclass
class Arm:
    name: str
    run: ArmFn
    # Free-form description of what makes this arm different — prompt version,
    # model, cache config. Recorded on the run so a result stays interpretable
    # after the code that produced it has moved on.
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmScores:
    name: str
    # metric -> per-example scores, in dataset order.
    per_metric: dict[str, list[float]] = field(default_factory=dict)
    responses: dict[str, Response] = field(default_factory=dict)
    failures: int = 0

    @property
    def total_cost_usd(self) -> Decimal:
        return sum((r.cost_usd for r in self.responses.values()), Decimal(0))

    @property
    def failure_rate(self) -> float:
        if not self.responses:
            return 1.0
        return self.failures / len(self.responses)

    def interval(self, metric: str, *, seed: int = 0) -> Interval:
        return bootstrap_ci(self.per_metric.get(metric, []), seed=seed)

    @property
    def cost_per_successful_task(self) -> Decimal | None:
        """The number that actually matters.

        A cheap arm that fails 40% of the time and triggers a retry on the
        expensive one costs more than going straight to the expensive one. Cost
        per token cannot show that; cost per *successful* task can.
        """
        successes = sum(
            1 for eid, r in self.responses.items() if r.ok and self._example_succeeded(eid)
        )
        if successes == 0:
            return None
        return self.total_cost_usd / successes

    def _example_succeeded(self, example_id: str) -> bool:
        """Success = scored 1.0 on the strictest metric available for it."""
        index = self._index.get(example_id)
        if index is None:
            return False
        for metric in ("exact_match", "schema_conformance", "contains", "json_valid"):
            scores = self.per_metric.get(metric)
            if scores and index < len(scores):
                return scores[index] >= 1.0
        f1 = self.per_metric.get("token_f1")
        if f1 and index < len(f1):
            return f1[index] >= 0.8
        return False

    _index: dict[str, int] = field(default_factory=dict)


@dataclass
class RunReport:
    run_id: str
    dataset: str
    started_at: datetime
    finished_at: datetime
    candidate: ArmScores
    baseline: ArmScores | None = None
    metric_intervals: dict[str, Interval] = field(default_factory=dict)
    metric_deltas: dict[str, Interval] = field(default_factory=dict)
    judge_results: list[PairwiseResult] = field(default_factory=list)
    judge_summary: dict[str, Any] = field(default_factory=dict)
    judge_delta: Interval | None = None
    calibration: CalibrationReport | None = None
    tolerance: float = 0.0

    @property
    def failure_rate(self) -> float:
        """The worst arm's share of failed calls.

        A run where most calls failed is not a measurement of quality, and both
        arms failing *equally* is the dangerous case: every metric reads 0.0, every
        delta reads 0.0, and the gate passes. Callers must check this before
        trusting `regressed`.
        """
        rates = [self.candidate.failure_rate]
        if self.baseline is not None:
            rates.append(self.baseline.failure_rate)
        return max(rates)

    def measured(self, *, max_failure_rate: float = 0.10) -> bool:
        return self.failure_rate <= max_failure_rate

    @property
    def regressed(self) -> bool:
        """The gate decision. Any metric whose whole interval is a drop fails."""
        return any(
            regression_detected(d, tolerance=self.tolerance) for d in self.metric_deltas.values()
        ) or (
            self.judge_delta is not None
            and regression_detected(self.judge_delta, tolerance=self.tolerance)
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "candidate": self.candidate.name,
            "baseline": self.baseline.name if self.baseline else None,
            "metrics": {k: v.as_json() for k, v in self.metric_intervals.items()},
            "deltas": {k: v.as_json() for k, v in self.metric_deltas.items()},
            "judge": self.judge_summary,
            "judge_delta": self.judge_delta.as_json() if self.judge_delta else None,
            "calibration": self.calibration.as_json() if self.calibration else None,
            "cost_usd": str(self.candidate.total_cost_usd),
            "cost_per_successful_task": (
                str(self.candidate.cost_per_successful_task)
                if self.candidate.cost_per_successful_task is not None
                else None
            ),
            "failures": self.candidate.failures,
            "failure_rate": self.failure_rate,
            "regressed": self.regressed,
        }

    def render(self) -> str:
        lines = [
            f"eval run {self.run_id}  dataset={self.dataset}",
            f"  candidate: {self.candidate.name}"
            + (f"   baseline: {self.baseline.name}" if self.baseline else ""),
            "",
        ]
        for metric, interval in sorted(self.metric_intervals.items()):
            line = f"  {metric:22} {interval}"
            delta = self.metric_deltas.get(metric)
            if delta is not None:
                verdict = (
                    "REGRESSION"
                    if regression_detected(delta, tolerance=self.tolerance)
                    else ("improved" if delta.excludes_zero else "no change")
                )
                # ASCII only: this text goes to CI logs and Windows consoles,
                # where a stray non-cp1252 glyph turns a passing gate into a
                # UnicodeEncodeError.
                line += (
                    f"   delta {delta.point:+.4f} [{delta.low:+.4f}, {delta.high:+.4f}] {verdict}"
                )
            lines.append(line)

        if self.judge_summary:
            s = self.judge_summary
            lines += [
                "",
                f"  judge: {s['candidate_wins']}W / {s['baseline_wins']}L / "
                f"{s['ties']}T / {s['inconsistent']} inconsistent "
                f"({s['errors']} errored)",
                f"  position-bias rate: {s['position_bias_rate']:.1%}",
            ]
            if self.judge_delta is not None:
                lines.append(f"  judge win rate: {self.judge_delta}")
        if self.calibration is not None:
            lines += ["", f"  {self.calibration}"]
            if not self.calibration.trustworthy:
                lines.append(
                    "  WARNING: judge agreement is below substantial - treat "
                    "judge-derived numbers as directional only."
                )

        if self.failure_rate > 0:
            lines += [
                "",
                f"  failed calls: {self.candidate.failures}/"
                f"{len(self.candidate.responses)} candidate"
                + (
                    f", {self.baseline.failures}/{len(self.baseline.responses)} baseline"
                    if self.baseline is not None
                    else ""
                ),
            ]

        lines += [
            "",
            f"  cost: ${self.candidate.total_cost_usd:.4f}"
            + (
                f"   per successful task: ${self.candidate.cost_per_successful_task:.6f}"
                if self.candidate.cost_per_successful_task is not None
                else ""
            ),
            "  gate: "
            + (
                # A run where the calls did not land has no verdict to give. Saying
                # "pass" here would be the single most misleading line in the tool.
                f"NOT MEASURED - {self.failure_rate:.0%} of calls failed"
                if not self.measured()
                else ("FAIL - regression detected" if self.regressed else "pass")
            ),
        ]
        return "\n".join(lines)


async def _run_arm(arm: Arm, dataset: Dataset, *, concurrency: int) -> ArmScores:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(example: Example) -> Response:
        async with semaphore:
            try:
                return await arm.run(example)
            except Exception as exc:  # noqa: BLE001 — one failure must not sink a run
                return Response(example.id, "", error=str(exc))

    responses = await asyncio.gather(*(one(e) for e in dataset))

    scores = ArmScores(name=arm.name)
    scores.responses = {r.example_id: r for r in responses}
    scores._index = {e.id: i for i, e in enumerate(dataset)}
    scores.failures = sum(1 for r in responses if not r.ok)

    for example, response in zip(dataset, responses, strict=True):
        for metric in example.metrics:
            scorer = SCORERS.get(metric)
            if scorer is None:
                continue
            # A failed call scores zero rather than being dropped. Dropping it
            # would let an arm improve its average by erroring on hard examples.
            value = 0.0 if not response.ok else float(scorer(response.text, example))
            scores.per_metric.setdefault(metric, []).append(value)

    return scores


async def run(
    dataset: Dataset,
    candidate: Arm,
    *,
    baseline: Arm | None = None,
    judge: PairwiseJudge | None = None,
    concurrency: int = 8,
    tolerance: float = 0.0,
    seed: int = 0,
    calibration: CalibrationReport | None = None,
) -> RunReport:
    started = datetime.now(UTC)
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    candidate_scores = await _run_arm(candidate, dataset, concurrency=concurrency)
    baseline_scores = (
        await _run_arm(baseline, dataset, concurrency=concurrency) if baseline else None
    )

    metric_intervals = {
        metric: bootstrap_ci(values, seed=seed)
        for metric, values in candidate_scores.per_metric.items()
    }

    metric_deltas: dict[str, Interval] = {}
    if baseline_scores is not None:
        for metric, values in candidate_scores.per_metric.items():
            base = baseline_scores.per_metric.get(metric)
            if base and len(base) == len(values):
                metric_deltas[metric] = paired_bootstrap_ci(values, base, seed=seed)

    judge_results: list[PairwiseResult] = []
    judge_summary: dict[str, Any] = {}
    judge_delta: Interval | None = None

    if judge is not None and baseline_scores is not None:
        open_ended = dataset.judged().examples
        judge_results = await _judge_all(
            judge, open_ended, candidate_scores, baseline_scores, concurrency=concurrency
        )
        if judge_results:
            judge_summary = summarize(judge_results)
            usable = [r for r in judge_results if r.is_usable]
            if usable:
                # 0.5 is the no-difference point for pairwise scoring, so the
                # interval is centred there to read as a delta like the others.
                judge_delta = bootstrap_ci([r.candidate_score - 0.5 for r in usable], seed=seed)

    return RunReport(
        run_id=run_id,
        dataset=str(dataset.source or "in-memory"),
        started_at=started,
        finished_at=datetime.now(UTC),
        candidate=candidate_scores,
        baseline=baseline_scores,
        metric_intervals=metric_intervals,
        metric_deltas=metric_deltas,
        judge_results=judge_results,
        judge_summary=judge_summary,
        judge_delta=judge_delta,
        calibration=calibration,
        tolerance=tolerance,
    )


async def _judge_all(
    judge: PairwiseJudge,
    examples: Sequence[Example],
    candidate: ArmScores,
    baseline: ArmScores,
    *,
    concurrency: int,
) -> list[PairwiseResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(example: Example) -> PairwiseResult | None:
        a = candidate.responses.get(example.id)
        b = baseline.responses.get(example.id)
        if a is None or b is None or not a.ok or not b.ok:
            return None
        async with semaphore:
            return await judge.compare(example, a.text, b.text)

    results = await asyncio.gather(*(one(e) for e in examples))
    return [r for r in results if r is not None]
