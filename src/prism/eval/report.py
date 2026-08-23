"""Rendering an eval run for humans who are about to be blocked by it.

A gate that only says "failed" gets overridden. One that shows which metric
moved, by how much, with what interval, and whether the judge behind the number
was ever calibrated gets acted on. This is the whole difference between a check
people trust and a check people disable.

Output is ASCII-only for the same reason the console renderer is: CI logs and
Windows consoles are cp1252, and a stray glyph turns a passing gate into a
UnicodeEncodeError that says nothing about quality.
"""

from __future__ import annotations

from .runner import RunReport
from .stats import regression_detected


def markdown_summary(report: RunReport) -> str:
    """Rendered on the Actions run page, so a failure explains itself.

    A gate that only says "failed" gets overridden; one that shows which metric
    moved, by how much, and with what interval gets acted on.
    """
    if not report.measured():
        return (
            "## Eval gate: NOT MEASURED\n\n"
            f"{report.failure_rate:.0%} of calls failed, so every metric reads zero "
            "for both arms and no comparison is meaningful. This is a broken run, "
            "not a passing one.\n"
        )

    verdict = "REGRESSION DETECTED" if report.regressed else "passed"
    lines = [
        f"## Eval gate: {verdict}",
        "",
        f"- candidate: `{report.candidate.name}`",
        f"- baseline: `{report.baseline.name if report.baseline else 'none'}`",
        f"- dataset: `{report.dataset}`",
        f"- tolerance: `{report.tolerance}`",
        "",
        "| metric | candidate | 95% CI | delta | verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric, interval in sorted(report.metric_intervals.items()):
        delta = report.metric_deltas.get(metric)
        if delta is None:
            lines.append(
                f"| {metric} | {interval.point:.4f} "
                f"| [{interval.low:.4f}, {interval.high:.4f}] | - | - |"
            )
            continue
        state = (
            "**regression**"
            if regression_detected(delta, tolerance=report.tolerance)
            else ("improved" if delta.excludes_zero else "no change")
        )
        lines.append(
            f"| {metric} | {interval.point:.4f} "
            f"| [{interval.low:.4f}, {interval.high:.4f}] "
            f"| {delta.point:+.4f} [{delta.low:+.4f}, {delta.high:+.4f}] | {state} |"
        )

    if report.judge_summary:
        js = report.judge_summary
        lines += [
            "",
            "### Judge",
            "",
            f"- {js['candidate_wins']}W / {js['baseline_wins']}L / {js['ties']}T "
            f"/ {js['inconsistent']} inconsistent ({js['errors']} errored)",
            f"- position-bias rate: {js['position_bias_rate']:.1%}",
        ]
        if report.calibration is not None:
            c = report.calibration
            lines.append(
                f"- calibration: kappa = {c.kappa:.3f} ({c.interpretation}) on n={c.n} human labels"
            )
            if not c.trustworthy:
                lines.append(
                    "- **judge agreement is below substantial - treat the judge "
                    "columns as directional only**"
                )
        else:
            lines.append(
                "- **uncalibrated: no human labels supplied, so these verdicts "
                "carry no known agreement rate**"
            )

    lines += [
        "",
        f"Cost: ${report.candidate.total_cost_usd:.4f}"
        + (
            f" (${report.candidate.cost_per_successful_task:.6f} per successful task)"
            if report.candidate.cost_per_successful_task is not None
            else ""
        ),
        "",
        "Intervals are paired bootstrap, 95%, fixed seed. The gate fails only "
        "when a metric's *entire* interval sits below tolerance.",
    ]
    return "\n".join(lines) + "\n"
