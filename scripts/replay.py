"""Three-configuration cache replay: no cache, prefix only, prefix + semantic.

    python scripts/replay.py --from-pairs datasets/cache_pairs.jsonl --threshold 0.91

Reports the *incremental* saving of the semantic layer over prefix caching — the
number this project exists to produce — alongside the false-hit exposure that
increment costs.

**The output is a simulation.** Provider cache state is not controllable from
outside, so replaying live three times would not isolate the variable and would
cost three times as much. The simulation runs real breakpoint placement, real
scope keys, the real embedder, and the real five-class cost model, and models
only prefix liveness (a TTL window) and semantic hits.

    --corpus       JSONL of recorded requests exported from the trace table
    --from-pairs   synthesize a corpus from a labelled pair file instead; each
                   equivalent pair becomes a repeat, each non-equivalent pair a
                   distinct request. Useful before there is production traffic,
                   and honest about being synthetic.
    --audit-out    write every semantic hit to a file for human false-hit labelling
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import UTC, datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from prism.caching.calibration import load_pairs  # noqa: E402
from prism.caching.embeddings import EmbeddingUnavailableError, build_embedder  # noqa: E402
from prism.caching.replay import ReplayRequest, audit_served, run  # noqa: E402
from prism.cost import TokenUsage  # noqa: E402

SYSTEM = "You are a careful assistant that answers precisely and briefly. " * 200


def synthesize(pairs_path: str, prompts_root: pathlib.Path) -> list[ReplayRequest]:
    """Turn labelled pairs into a request stream.

    Equivalent pairs become repeat traffic (the semantic layer's opportunity);
    non-equivalent pairs become distinct requests (its exposure). The mix is a
    stand-in for a real corpus, not a claim about one.
    """
    pairs = load_pairs(pairs_path)
    start = datetime(2026, 8, 23, 12, 0, 0, tzinfo=UTC)
    corpus: list[ReplayRequest] = []
    for i, pair in enumerate(pairs):
        for j, query in enumerate((pair.query_a, pair.query_b)):
            corpus.append(
                ReplayRequest(
                    request_id=f"p{i}-{j}",
                    tenant_id="tenant-1",
                    timestamp=start + timedelta(seconds=10 * len(corpus)),
                    body={
                        "model": "claude-opus-5",
                        "max_tokens": 1024,
                        "system": [{"type": "text", "text": SYSTEM}],
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": query}]}
                        ],
                    },
                    response={
                        "content": [{"type": "text", "text": f"answer to: {query}"}],
                        "stop_reason": "end_turn",
                    },
                    usage=TokenUsage(input_tokens=3000, output_tokens=400),
                    prompt_version="replay@v1",
                )
            )
    return corpus


def write_synthetic_registry(root: pathlib.Path) -> None:
    """The scope decision needs a registry the corpus's prompt actually matches."""
    (root / "replay").mkdir(parents=True, exist_ok=True)
    (root / "replay" / "v1.md").write_text(SYSTEM, encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"replay": {"active": "v1"}}, indent=2), encoding="utf-8"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus")
    parser.add_argument("--from-pairs")
    parser.add_argument("--threshold", type=float, default=0.91)
    parser.add_argument("--ttl", default="5m", choices=["5m", "1h"])
    parser.add_argument("--embedder", default="local", choices=["local", "hashing"])
    parser.add_argument("--prompts", default="prompts")
    parser.add_argument("--audit-out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.corpus and not args.from_pairs:
        parser.error("one of --corpus or --from-pairs is required")

    prompts_root = pathlib.Path(args.prompts)
    if args.from_pairs:
        import tempfile

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="prism-replay-"))
        write_synthetic_registry(tmp)
        prompts_root = tmp
        corpus = synthesize(args.from_pairs, tmp)
        print(
            f"SYNTHETIC corpus: {len(corpus)} requests built from "
            f"{args.from_pairs}. Not production traffic.\n",
            file=sys.stderr,
        )
    else:
        corpus = load_corpus(args.corpus)

    try:
        embedder = build_embedder(args.embedder)
    except EmbeddingUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = run(
        corpus,
        embedder,
        threshold=args.threshold,
        ttl=args.ttl,
        prompts_root=str(prompts_root),
    )

    print(json.dumps(report.as_json(), indent=2) if args.json else report.render())

    if args.audit_out:
        rows = audit_served(report, corpus)
        pathlib.Path(args.audit_out).write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        print(
            f"\n{len(rows)} semantic hits written to {args.audit_out}. Label the "
            "`false_hit` field by hand: an exposure figure nobody inspected is an "
            "assumption, not a measurement.",
            file=sys.stderr,
        )
    return 0


def load_corpus(path: str) -> list[ReplayRequest]:
    out: list[ReplayRequest] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        raw = json.loads(line)
        usage = raw.get("usage") or {}
        out.append(
            ReplayRequest(
                request_id=raw["request_id"],
                tenant_id=raw["tenant_id"],
                timestamp=datetime.fromisoformat(raw["timestamp"]),
                body=raw["body"],
                response=raw.get("response") or {},
                usage=TokenUsage(
                    input_tokens=usage.get("input_tokens", 0),
                    cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                    cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                ),
                prompt_version=raw.get("prompt_version"),
            )
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
