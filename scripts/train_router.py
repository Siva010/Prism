"""Train the difficulty router from eval outcomes.

    python scripts/train_router.py --outcomes data/router_outcomes.jsonl

Outcomes come from the eval harness: run the golden set on the cheap tier and
the expensive one, score both, and label each example with whether the cheap
answer was acceptable. `--demo` synthesizes a set so the pipeline can be
exercised before real eval runs exist, and says so loudly.

Exits non-zero when the trained router does not beat a coin flip — a router with
AUC ~0.5 is worse than picking one tier, because it adds latency and a moving
part to reach the same answer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from prism.eval.dataset import load  # noqa: E402
from prism.routing.economics import break_even_table  # noqa: E402
from prism.routing.features import extract  # noqa: E402
from prism.routing.model import (  # noqa: E402
    DifficultyRouter,
    RouterReport,
    RouterUnavailableError,
    TrainingRow,
)
from prism.routing.training import load_outcomes  # noqa: E402


def demo_rows(n: int, seed: int) -> list[TrainingRow]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        easy = rng.random() < 0.5
        text = (
            f"What is the capital of region {i}?"
            if easy
            else "Prove step by step why this invariant holds, analyse the "
            f"tradeoff, and design an alternative approach. Case {i}."
        )
        succeeded = easy if rng.random() > 0.15 else not easy
        rows.append(
            TrainingRow(
                f"demo-{i}",
                extract(
                    {
                        "model": "claude-opus-5",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
                    }
                ),
                succeeded,
            )
        )
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes")
    parser.add_argument("--dataset", default="datasets/golden/v1.jsonl")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-size", type=int, default=400)
    parser.add_argument("--out", default="data/router.joblib")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--project-dimensions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.demo:
        print(
            "SYNTHETIC training data. This exercises the pipeline; it says "
            "nothing about how the router would perform on real traffic.",
            file=sys.stderr,
        )
        rows = demo_rows(args.demo_size, args.seed)
    elif args.outcomes:
        outcomes = load_outcomes(args.outcomes)
        from prism.routing.training import build

        dataset = load(args.dataset).split("router_train")
        holdout = load(args.dataset).split("test")
        # The guard runs before fitting, not as a review step somebody skips.
        rows = build(dataset, outcomes, holdout=holdout)
    else:
        parser.error("one of --outcomes or --demo is required")

    if len(rows) < 50:
        print(
            f"WARNING: only {len(rows)} labelled rows. A router fitted on this "
            "little will not generalise, and its held-out numbers will be noise.",
            file=sys.stderr,
        )

    rng = random.Random(args.seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - args.test_fraction))
    train, test = shuffled[:cut], shuffled[cut:]

    router = DifficultyRouter(project_dimensions=args.project_dimensions, seed=args.seed)
    try:
        router.fit(train)
    except (RouterUnavailableError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    auc, brier, curve = router.evaluate(test)
    report = RouterReport(
        n_train=len(train),
        n_test=len(test),
        auc=auc,
        brier=brier,
        curve=curve,
        coefficients=router.coefficients(),
        base_rate=sum(r.cheap_succeeded for r in test) / len(test) if test else 0.0,
    )

    print(json.dumps(report.as_json(), indent=2) if args.json else report.render())

    print("\n  break-even thresholds this router will be compared against:")
    for row in break_even_table():
        print(
            f"    {row['cheap']:<20} -> {row['expensive']:<18} "
            f"route down when P > {row['threshold']:.3f}"
        )

    if not report.beats_base_rate:
        return 1

    router.save(args.out)
    print(f"\n  saved to {args.out} (coefficients alongside as .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
