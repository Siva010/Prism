"""Run the evaluation suite.

    python scripts/eval.py --dataset datasets/golden/v1.jsonl --candidate v2 --baseline v1

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
from prism.eval.runner import run  # noqa: E402


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
    p.add_argument("--dataset", default="datasets/golden/v1.jsonl")
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
    return p


async def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    dataset = load(args.dataset).split(args.split)
    if not len(dataset):
        print(f"no examples in split {args.split!r} of {args.dataset}", file=sys.stderr)
        return 2

    model = args.model or settings.default_model
    candidate = GatewayArm(
        args.candidate,
        base_url=args.gateway_url,
        api_key=args.api_key,
        model=model,
        prompt_version=args.candidate,
    )
    baseline = (
        GatewayArm(
            args.baseline,
            base_url=args.gateway_url,
            api_key=args.api_key,
            model=args.baseline_model or model,
            prompt_version=args.baseline,
        )
        if args.baseline
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
        report.calibration = calibrate(
            report.judge_results,
            load_human_labels(args.human_labels),
            judge_model=judge_model or "",
        )

    print(json.dumps(report.as_json(), indent=2) if args.json else report.render())

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
