"""Persisting eval runs.

Results live in Postgres next to the traces, which is the point of keeping one
database: a run's per-example results carry the `trace_id` of the request that
produced them, so "why did this example regress?" is a join rather than an
archaeology exercise across two systems.

Nothing here stores a point estimate without its interval.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .judge import PairwiseResult
from .runner import ArmScores, RunReport

_INSERT_RUN = text("""
INSERT INTO eval_runs (
    run_id, created_at, finished_at, dataset, dataset_size,
    candidate_name, candidate_config, baseline_name, baseline_config,
    metrics, deltas, judge_model, judge_summary, judge_delta, calibration,
    cost_usd, cost_per_success, failures, regressed, tolerance, git_sha, ci_run_url
) VALUES (
    :run_id, :created_at, :finished_at, :dataset, :dataset_size,
    :candidate_name, CAST(:candidate_config AS JSONB), :baseline_name,
    CAST(:baseline_config AS JSONB),
    CAST(:metrics AS JSONB), CAST(:deltas AS JSONB), :judge_model,
    CAST(:judge_summary AS JSONB), CAST(:judge_delta AS JSONB),
    CAST(:calibration AS JSONB),
    :cost_usd, :cost_per_success, :failures, :regressed, :tolerance,
    :git_sha, :ci_run_url
)
ON CONFLICT (run_id) DO NOTHING
""")

_INSERT_RESULT = text("""
INSERT INTO eval_results (
    run_id, example_id, arm, response_text, model, cost_usd, latency_ms,
    trace_id, error, scores, judge_verdict, judge_reason,
    judge_raw_forward, judge_raw_reverse
) VALUES (
    :run_id, :example_id, :arm, :response_text, :model, :cost_usd, :latency_ms,
    :trace_id, :error, CAST(:scores AS JSONB), :judge_verdict, :judge_reason,
    :judge_raw_forward, :judge_raw_reverse
)
ON CONFLICT (run_id, example_id, arm) DO NOTHING
""")


def _json(value: Any) -> str:
    import json

    return json.dumps(value, default=str)


async def save_run(
    session: AsyncSession,
    report: RunReport,
    *,
    example_order: list[str],
    judge_model: str | None = None,
    git_sha: str | None = None,
    ci_run_url: str | None = None,
) -> str:
    await session.execute(
        _INSERT_RUN,
        {
            "run_id": report.run_id,
            "created_at": report.started_at,
            "finished_at": report.finished_at,
            "dataset": report.dataset,
            "dataset_size": len(example_order),
            "candidate_name": report.candidate.name,
            "candidate_config": _json({}),
            "baseline_name": report.baseline.name if report.baseline else None,
            "baseline_config": _json({}),
            "metrics": _json({k: v.as_json() for k, v in report.metric_intervals.items()}),
            "deltas": _json({k: v.as_json() for k, v in report.metric_deltas.items()}),
            "judge_model": judge_model,
            "judge_summary": _json(report.judge_summary),
            "judge_delta": _json(report.judge_delta.as_json()) if report.judge_delta else None,
            "calibration": _json(report.calibration.as_json()) if report.calibration else None,
            "cost_usd": report.candidate.total_cost_usd,
            "cost_per_success": report.candidate.cost_per_successful_task,
            "failures": report.candidate.failures,
            "regressed": report.regressed,
            "tolerance": report.tolerance,
            "git_sha": git_sha,
            "ci_run_url": ci_run_url,
        },
    )

    judged = {r.example_id: r for r in report.judge_results}
    for arm_label, scores in (
        ("candidate", report.candidate),
        ("baseline", report.baseline),
    ):
        if scores is None:
            continue
        await _save_arm(
            session,
            report.run_id,
            arm_label,
            scores,
            example_order,
            judged if arm_label == "candidate" else {},
        )

    await session.commit()
    return report.run_id


async def _save_arm(
    session: AsyncSession,
    run_id: str,
    arm_label: str,
    scores: ArmScores,
    example_order: list[str],
    judged: dict[str, PairwiseResult],
) -> None:
    for index, example_id in enumerate(example_order):
        response = scores.responses.get(example_id)
        if response is None:
            continue
        per_example = {
            metric: values[index]
            for metric, values in scores.per_metric.items()
            if index < len(values)
        }
        verdict = judged.get(example_id)
        await session.execute(
            _INSERT_RESULT,
            {
                "run_id": run_id,
                "example_id": example_id,
                "arm": arm_label,
                "response_text": response.text,
                "model": response.model,
                "cost_usd": response.cost_usd or Decimal(0),
                "latency_ms": response.latency_ms,
                "trace_id": response.trace_id,
                "error": response.error,
                "scores": _json(per_example),
                "judge_verdict": str(verdict.verdict) if verdict else None,
                "judge_reason": verdict.reason if verdict else None,
                "judge_raw_forward": verdict.first_order if verdict else None,
                "judge_raw_reverse": verdict.second_order if verdict else None,
            },
        )


async def latest_run(
    session: AsyncSession, dataset: str, *, candidate: str | None = None
) -> dict[str, Any] | None:
    """The most recent run for a dataset — the CI gate's implicit baseline."""
    stmt = text("""
        SELECT run_id, created_at, candidate_name, metrics, deltas, regressed,
               cost_usd, cost_per_success, calibration
        FROM eval_runs
        WHERE dataset = :dataset
          AND (:candidate IS NULL OR candidate_name = :candidate)
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = (
        (await session.execute(stmt, {"dataset": dataset, "candidate": candidate}))
        .mappings()
        .first()
    )
    return dict(row) if row else None


async def save_human_labels(
    session: AsyncSession, labels: list[dict[str, Any]], *, source_run_id: str | None
) -> int:
    stmt = text("""
        INSERT INTO human_labels (example_id, label, annotator, notes, source_run_id)
        VALUES (:example_id, :label, :annotator, :notes, :source_run_id)
        ON CONFLICT (example_id, annotator, source_run_id) DO UPDATE
            SET label = EXCLUDED.label, notes = EXCLUDED.notes
    """)
    for label in labels:
        await session.execute(stmt, {**label, "source_run_id": source_run_id})
    await session.commit()
    return len(labels)
