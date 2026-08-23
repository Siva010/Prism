"""Local token estimation, and the machinery to find out how wrong it is.

A ground-truth oracle exists — the token counting endpoint returns exact counts —
but calling it adds a round trip to every request, which is unacceptable on the
hot path for a component whose entire job is to save money. The correct design is
a fast local estimator *calibrated offline against* the oracle, with periodic
drift checks.

So this module is deliberately two things:

1. A cheap estimator, used synchronously wherever a token count is needed before
   dispatch (breakpoint placement in week 7, rate-limit reservation in week 10).
2. A calibration harness that measures the estimator's error against the oracle
   and reports the *distribution*, not a single accuracy number.

The distribution is what matters, because the two consumers care about opposite
tails. Breakpoint placement over-estimating means placing a breakpoint on a block
too small to cache — wasted, but harmless. Rate-limit reservation under-estimating
means over-committing a bucket and tripping a 429. An estimator described only by
its mean error hides both.

**This estimator is uncalibrated until someone runs `calibrate()` against a real
key.** `EstimatorReport.calibrated` says so, and nothing here pretends otherwise.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

# Rough characters-per-token for English prose under the Claude tokenizer.
# Punctuation and whitespace tokenize denser than letters, so the estimator
# counts them separately rather than dividing the whole string by one constant.
_CHARS_PER_TOKEN = 3.6
_WHITESPACE_RUN = re.compile(r"\s+")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")

# Per-message and per-block framing the API adds around content.
_MESSAGE_OVERHEAD_TOKENS = 3
_BLOCK_OVERHEAD_TOKENS = 2
_TOOL_OVERHEAD_TOKENS = 8

# An image costs tokens proportional to its area; without decoding it, base64
# length is the only signal available. Deliberately generous: under-counting
# images is the failure that trips a rate limit.
_IMAGE_BASE64_CHARS_PER_TOKEN = 750


def estimate_text(text: str) -> int:
    """Estimate tokens in a plain string."""
    if not text:
        return 0
    # Runs of whitespace collapse into roughly one token regardless of length.
    collapsed = _WHITESPACE_RUN.sub(" ", text)
    # Non-ASCII characters typically cost more than one token each.
    non_ascii = len(_NON_ASCII.findall(collapsed))
    ascii_chars = len(collapsed) - non_ascii
    return max(1, math.ceil(ascii_chars / _CHARS_PER_TOKEN) + non_ascii * 2)


def estimate_content_block(block: dict[str, Any] | str) -> int:
    """Estimate one Anthropic content block."""
    if isinstance(block, str):
        return estimate_text(block)

    kind = block.get("type")
    if kind == "text":
        return _BLOCK_OVERHEAD_TOKENS + estimate_text(block.get("text", ""))
    if kind == "thinking":
        return _BLOCK_OVERHEAD_TOKENS + estimate_text(block.get("thinking", ""))
    if kind == "image":
        source = block.get("source") or {}
        if source.get("type") == "base64":
            return _BLOCK_OVERHEAD_TOKENS + math.ceil(
                len(source.get("data", "")) / _IMAGE_BASE64_CHARS_PER_TOKEN
            )
        # A URL source is fetched server-side; its size is unknowable from here.
        # 1500 is a mid-sized image, and being wrong in the cheap direction here
        # would silently under-reserve.
        return _BLOCK_OVERHEAD_TOKENS + 1500
    if kind == "tool_use":
        return (
            _BLOCK_OVERHEAD_TOKENS
            + estimate_text(block.get("name", ""))
            + estimate_text(json.dumps(block.get("input") or {}, separators=(",", ":")))
        )
    if kind == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            inner = estimate_text(content)
        elif isinstance(content, list):
            inner = sum(estimate_content_block(b) for b in content)
        else:
            inner = 0
        return _BLOCK_OVERHEAD_TOKENS + inner
    # Unknown block type: estimate from its serialized form rather than skipping
    # it. Zero would be the one answer guaranteed to be wrong.
    return _BLOCK_OVERHEAD_TOKENS + estimate_text(json.dumps(block, separators=(",", ":")))


def estimate_blocks(blocks: Any) -> int:
    if blocks is None:
        return 0
    if isinstance(blocks, str):
        return estimate_text(blocks)
    if isinstance(blocks, list):
        return sum(estimate_content_block(b) for b in blocks)
    return estimate_content_block(blocks)


def estimate_tool(tool: dict[str, Any]) -> int:
    return (
        _TOOL_OVERHEAD_TOKENS
        + estimate_text(tool.get("name", ""))
        + estimate_text(tool.get("description", ""))
        + estimate_text(json.dumps(tool.get("input_schema") or {}, separators=(",", ":")))
    )


def estimate_request(body: dict[str, Any]) -> int:
    """Estimate the input tokens of a full Anthropic Messages request."""
    total = 0
    for tool in body.get("tools") or []:
        total += estimate_tool(tool)
    total += estimate_blocks(body.get("system"))
    for message in body.get("messages") or []:
        total += _MESSAGE_OVERHEAD_TOKENS + estimate_blocks(message.get("content"))
    return total


# --- calibration ---------------------------------------------------------


@dataclass(frozen=True)
class EstimatorSample:
    estimated: int
    actual: int

    @property
    def error(self) -> int:
        return self.estimated - self.actual

    @property
    def relative_error(self) -> float:
        return self.error / self.actual if self.actual else 0.0


@dataclass
class EstimatorReport:
    samples: list[EstimatorSample] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        return len(self.samples) >= 20

    def percentiles(self) -> dict[str, float]:
        """Relative-error distribution, both tails.

        The tails are the point. Breakpoint placement can absorb over-estimation;
        rate-limit reservation cannot absorb under-estimation. A mean would hide
        which one is happening.
        """
        if not self.samples:
            return {}
        errors = sorted(s.relative_error for s in self.samples)

        def pct(p: float) -> float:
            if len(errors) == 1:
                return errors[0]
            index = p * (len(errors) - 1)
            low, high = math.floor(index), math.ceil(index)
            return errors[low] + (errors[high] - errors[low]) * (index - low)

        return {
            "p01": pct(0.01),
            "p05": pct(0.05),
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "mean": sum(errors) / len(errors),
            "max_under": min(errors),
            "max_over": max(errors),
        }

    def as_json(self) -> dict[str, Any]:
        return {
            "n": len(self.samples),
            "calibrated": self.calibrated,
            "relative_error": self.percentiles(),
        }

    def __str__(self) -> str:
        if not self.samples:
            return "estimator: uncalibrated (no samples)"
        p = self.percentiles()
        return (
            f"estimator on n={len(self.samples)}: "
            f"median {p['p50']:+.1%}, "
            f"p05 {p['p05']:+.1%}, p95 {p['p95']:+.1%}, "
            f"worst under {p['max_under']:+.1%}, worst over {p['max_over']:+.1%}"
        )


async def calibrate(bodies: list[dict[str, Any]], count_tokens: Any) -> EstimatorReport:
    """Compare the local estimator against the counting endpoint.

    `count_tokens` is an async callable taking a request body and returning the
    exact input-token count — the provider's oracle. Run offline, never on the
    hot path.
    """
    report = EstimatorReport()
    for body in bodies:
        try:
            actual = await count_tokens(body)
        except Exception:  # noqa: BLE001 — a failed oracle call is a lost sample
            continue
        report.samples.append(EstimatorSample(estimate_request(body), int(actual)))
    return report
