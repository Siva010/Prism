"""Measuring the measuring instrument.

Most people using LLM-as-judge trust it. This module is the argument that you
should not: it takes a set of comparisons a human has labelled and asks whether
the judge agrees with the human any better than chance would.

The output is Cohen's kappa. It is reported next to every judge-derived number
in this project, because a win rate produced by an instrument with κ = 0.15 is
not a measurement — it is a number with a decimal point.

Two deliberate details:

* **Human labels come from the same comparisons the judge saw**, not from a
  separate rating exercise. Comparing a judge's pairwise verdicts against
  independently-collected absolute ratings measures the mismatch between two
  tasks, not the judge's accuracy.
* **``inconsistent`` folds into ``tie`` before agreement is computed.** The
  human never had an order to be biased by, so they have no corresponding
  category, and scoring against a label the human could not produce would
  depress kappa for the wrong reason. The position-bias rate is reported
  separately instead.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .judge import PairwiseResult, Verdict, position_bias_rate
from .stats import Interval, cohens_kappa, interpret_kappa, wilson_interval

# What a human is asked to choose between. No "inconsistent" — that category
# only exists because the judge sees two orderings.
HUMAN_LABELS = ("A", "B", "tie")


@dataclass(frozen=True)
class HumanLabel:
    example_id: str
    label: str  # "A" (candidate better), "B" (baseline better), or "tie"
    annotator: str = "human"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.label not in HUMAN_LABELS:
            raise ValueError(
                f"{self.example_id}: label must be one of {HUMAN_LABELS}, got {self.label!r}"
            )


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    kappa: float
    raw_agreement: float
    interpretation: str
    position_bias_rate: float
    confusion: dict[str, dict[str, int]]
    judge_model: str = ""

    @property
    def trustworthy(self) -> bool:
        """Substantial agreement or better (Landis & Koch).

        A threshold, not a verdict. Below it, judge-derived numbers should be
        reported with the kappa attached and treated as directional at best.
        """
        return self.kappa >= 0.60

    def as_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "cohens_kappa": self.kappa,
            "raw_agreement": self.raw_agreement,
            "interpretation": self.interpretation,
            "position_bias_rate": self.position_bias_rate,
            "confusion": self.confusion,
            "judge_model": self.judge_model,
            "trustworthy": self.trustworthy,
        }

    def __str__(self) -> str:
        return (
            f"judge calibration on n={self.n}: "
            f"kappa = {self.kappa:.3f} ({self.interpretation}), "
            f"raw agreement {self.raw_agreement:.1%}, "
            f"position-bias rate {self.position_bias_rate:.1%}"
        )


def _collapse(verdict: Verdict) -> str:
    """Map a judge verdict onto the label set a human could have produced."""
    if verdict is Verdict.A:
        return "A"
    if verdict is Verdict.B:
        return "B"
    return "tie"  # TIE and INCONSISTENT both read as "no difference detected"


def calibrate(
    judge_results: Sequence[PairwiseResult],
    human_labels: Sequence[HumanLabel],
    *,
    judge_model: str = "",
) -> CalibrationReport:
    """Cohen's kappa between the judge and the humans, on shared examples."""
    by_id = {r.example_id: r for r in judge_results if r.is_usable}
    paired = [(by_id[h.example_id], h) for h in human_labels if h.example_id in by_id]

    if not paired:
        raise ValueError(
            "no overlap between judge results and human labels — the humans must "
            "label the same comparisons the judge ran"
        )

    judge_side = [_collapse(r.verdict) for r, _ in paired]
    human_side = [h.label for _, h in paired]

    confusion: dict[str, dict[str, int]] = {h: {j: 0 for j in HUMAN_LABELS} for h in HUMAN_LABELS}
    for judged, human in zip(judge_side, human_side, strict=True):
        confusion[human][judged] += 1

    agreements = sum(1 for j, h in zip(judge_side, human_side, strict=True) if j == h)
    kappa = cohens_kappa(judge_side, human_side, labels=HUMAN_LABELS)

    return CalibrationReport(
        n=len(paired),
        kappa=kappa,
        raw_agreement=agreements / len(paired),
        interpretation=interpret_kappa(kappa),
        position_bias_rate=position_bias_rate([r for r, _ in paired]),
        confusion=confusion,
        judge_model=judge_model,
    )


def agreement_interval(report: CalibrationReport, *, confidence: float = 0.95) -> Interval:
    """CI on raw agreement.

    ~100 labelled examples is the usual budget, and at n=100 an agreement rate
    carries roughly ±10 points of uncertainty. Reporting it bare invites a
    precision the sample size does not support.
    """
    successes = round(report.raw_agreement * report.n)
    return wilson_interval(successes, report.n, confidence=confidence)


def load_human_labels(path: str | Path) -> list[HumanLabel]:
    """Read JSONL human labels: {"example_id": ..., "label": "A"|"B"|"tie"}."""
    path = Path(path)
    labels: list[HumanLabel] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
                labels.append(
                    HumanLabel(
                        example_id=raw["example_id"],
                        label=raw["label"],
                        annotator=raw.get("annotator", "human"),
                        notes=raw.get("notes", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return labels


def write_labelling_sheet(
    judge_results: Sequence[PairwiseResult],
    responses: dict[str, tuple[str, str]],
    path: str | Path,
    *,
    limit: int = 100,
) -> int:
    """Emit comparisons for a human to label, with the judge's answer withheld.

    Showing the annotator what the judge thought would anchor them to it, and
    the resulting kappa would measure suggestibility rather than agreement.
    """
    path = Path(path)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for result in judge_results[:limit]:
            if not result.is_usable or result.example_id not in responses:
                continue
            candidate, baseline = responses[result.example_id]
            handle.write(
                json.dumps(
                    {
                        "example_id": result.example_id,
                        "response_A": candidate,
                        "response_B": baseline,
                        "label": None,  # fill in: "A" | "B" | "tie"
                        "notes": "",
                    }
                )
                + "\n"
            )
            written += 1
    return written
