"""Run the evaluation suite.

    python scripts/eval.py --dataset datasets/golden/v2.jsonl --candidate v2 --baseline v1

Exits non-zero when a regression is detected, which is the whole interface the
week-6 CI gate needs. Everything else — where results are stored, how the judge
is configured — is settings, not arguments to the gate.

    --split         which split to score (default: test)
    --candidate     prompt version for the candidate arm
    --baseline      prompt version for the baseline arm; omit for a single-arm run
    --model         model the arms request (default: settings.default_model)
    --tolerance     quality drop to accept, e.g. 0.05 for a cheaper tier
    --no-judge      skip pairwise judging even for open-ended examples
    --json          emit the report as JSON instead of text
    --no-store      do not write the run to Postgres
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from prism.config import get_settings  # noqa: E402
from prism.db.engine import dispose_engine, init_engine, session_scope  # noqa: E402
from prism.eval import store  # noqa: E402
from prism.eval.calibration import calibrate, load_human_labels  # noqa: E402
from prism.eval.clients import (  # noqa: E402
    GatewayArm,
    JudgeConfigurationError,
    OpenAICompatibleJudge,
)
from prism.eval.dataset import load  # noqa: E402
from prism.eval.judge import PairwiseJudge  # noqa: E402
from prism.eval.report import markdown_summary  # noqa: E402
from prism.eval.runner import run  # noqa: E402
from prism.prompts import PromptError, Registry  # noqa: E402


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _ci_run_url() -> str | None:
    """Link the stored run back to the Actions run that produced it."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not (server and repo and run_id):
        return None
    return f"{server}/{repo}/actions/runs/{run_id}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Prism evaluation suite.")
    p.add_argument("--dataset", default="datasets/golden/v2.jsonl")
    p.add_argument("--split", default="test")
    p.add_argument("--candidate", default="candidate")
    p.add_argument("--baseline", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--baseline-model", default=None)
    p.add_argument("--tolerance", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--human-labels", default=None)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--gateway-url", default=os.environ.get("PRISM_URL", "http://localhost:8000"))
    p.add_argument("--api-key", default=os.environ.get("PRISM_API_KEY", ""))
    p.add_argument("--prompts", default="prompts", help="prompt registry root")
    p.add_argument(
        "--auto",
        action="store_true",
        help="gate whatever the registry says is pending (candidate = newest "
        "version, baseline = active). Exits 0 with no run when nothing is pending.",
    )
    p.add_argument("--summary-file", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    p.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.10,
        help="abort rather than report a verdict when more than this share of "
        "calls failed. A gateway that is down makes both arms score zero, "
        "which otherwise looks like a clean pass.",
    )
    return p


async def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    dataset = load(args.dataset).split(args.split)
    if not len(dataset):
        print(f"no examples in split {args.split!r} of {args.dataset}", file=sys.stderr)
        return 2

    model = args.model or settings.default_model
    registry = Registry(args.prompts)

    candidate_ref, baseline_ref = args.candidate, args.baseline
    candidate_text = baseline_text = None

    if args.auto:
        pending = registry.all_pending()
        if not pending:
            # Nothing to gate. Passing here is correct: a gate that invented a
            # comparison would burn a full eval run on every unrelated commit.
            print("no pending prompt versions - nothing to gate.")
            return 0
        if len(pending) > 1:
            print(
                f"ERROR: {len(pending)} prompts have pending versions "
                f"({', '.join(sorted(pending))}). Gate them one at a time so a "
                "regression can be attributed to a single change.",
                file=sys.stderr,
            )
            return 2
        _name, (cand, base) = next(iter(pending.items()))
        candidate_ref, baseline_ref = cand.ref, base.ref
        candidate_text, baseline_text = cand.text, base.text
        print(f"gating {cand.ref} against {base.ref}")
    else:
        try:
            if "@" in str(candidate_ref):
                prompt = registry.resolve(candidate_ref)
                candidate_ref, candidate_text = prompt.ref, prompt.text
            if baseline_ref and "@" in str(baseline_ref):
                prompt = registry.resolve(baseline_ref)
                baseline_ref, baseline_text = prompt.ref, prompt.text
        except PromptError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    candidate = GatewayArm(
        candidate_ref,
        base_url=args.gateway_url,
        api_key=args.api_key,
        model=model,
        prompt_version=candidate_ref,
        system_prompt=candidate_text,
    )
    baseline = (
        GatewayArm(
            baseline_ref,
            base_url=args.gateway_url,
            api_key=args.api_key,
            model=args.baseline_model or model,
            prompt_version=baseline_ref,
            system_prompt=baseline_text,
        )
        if baseline_ref
        else None
    )

    judge = None
    judge_model = None
    if not args.no_judge and baseline is not None and dataset.judged().examples:
        judge_key = os.environ.get("PRISM_JUDGE_API_KEY", "")
        judge_model = os.environ.get("PRISM_JUDGE_MODEL", "")
        if not judge_key or not judge_model:
            # Loud, not silent. A run that quietly skipped the judge would report
            # only programmatic metrics while looking like a full evaluation.
            print(
                "WARNING: PRISM_JUDGE_MODEL / PRISM_JUDGE_API_KEY unset — "
                f"skipping {len(dataset.judged())} open-ended examples.",
                file=sys.stderr,
            )
        else:
            try:
                judge = PairwiseJudge(
                    OpenAICompatibleJudge(
                        judge_model,
                        api_key=judge_key,
                        base_url=os.environ.get(
                            "PRISM_JUDGE_BASE_URL", "https://api.openai.com/v1"
                        ),
                    )
                )
            except JudgeConfigurationError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2

    report = await run(
        dataset,
        candidate.as_arm(),
        baseline=baseline.as_arm() if baseline else None,
        judge=judge,
        concurrency=args.concurrency,
        tolerance=args.tolerance,
        seed=args.seed,
    )

    if args.human_labels and report.judge_results:
        labels = (
            load_human_labels(args.human_labels)
            if pathlib.Path(args.human_labels).is_file()
            else []
        )
        if labels:
            report.calibration = calibrate(
                report.judge_results, labels, judge_model=judge_model or ""
            )
        else:
            # Loud, and not fatal. An uncalibrated judge still produces verdicts;
            # what it does not produce is a known agreement rate, and the summary
            # says so rather than presenting the win rate as measured.
            print(
                f"WARNING: no human labels in {args.human_labels} - judge verdicts "
                "will be reported without a calibration figure.",
                file=sys.stderr,
            )

    print(json.dumps(report.as_json(), indent=2) if args.json else report.render())

    if not report.measured(max_failure_rate=args.max_failure_rate):
        # Exit 2, not 1: this is "the gate could not run", which is a different
        # thing from "quality regressed" and needs a different fix.
        print(
            f"\nERROR: {report.failure_rate:.0%} of calls failed "
            f"(limit {args.max_failure_rate:.0%}). Nothing was measured, so no "
            "verdict is reported. Check that the gateway at "
            f"{args.gateway_url} is running and the API key is valid.",
            file=sys.stderr,
        )
        return 2

    if args.summary_file:
        pathlib.Path(args.summary_file).write_text(markdown_summary(report), encoding="utf-8")

    if not args.no_store:
        init_engine(settings.database_url)
        try:
            async with session_scope() as session:
                await store.save_run(
                    session,
                    report,
                    example_order=[e.id for e in dataset],
                    judge_model=judge_model,
                    git_sha=git_sha(),
                    ci_run_url=_ci_run_url(),
                )
        except Exception as exc:  # noqa: BLE001
            # A storage failure must not turn a passing gate into a failing one.
            print(f"WARNING: could not persist run: {exc}", file=sys.stderr)
        finally:
            await dispose_engine()

    await candidate.aclose()
    if baseline:
        await baseline.aclose()

    return 1 if report.regressed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
