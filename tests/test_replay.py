"""The three-configuration replay, and the incremental number it exists to produce."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from prism.caching.embeddings import HashingEmbedder
from prism.caching.replay import ReplayRequest, audit_served, replay
from prism.cost import TokenUsage

# Long enough to clear the 1024-token floor, so a breakpoint is actually placed.
BIG_SYSTEM = "You are a careful assistant that answers precisely. " * 300


@pytest.fixture
def prompts(tmp_path: Path) -> str:
    """A registry whose v1 is the exact system prompt the corpus sends.

    The scope decision requires a byte-for-byte match against a versioned
    artifact before it will mark anything shared, so a replay corpus has to be
    paired with the registry it claims to come from.
    """
    (tmp_path / "assistant").mkdir()
    (tmp_path / "assistant" / "v1.md").write_text(BIG_SYSTEM, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"assistant": {"active": "v1"}}), encoding="utf-8"
    )
    return str(tmp_path)


def request(
    n: int,
    *,
    question: str | None = None,
    at: datetime | None = None,
    tenant: str = "t1",
    system: str | None = BIG_SYSTEM,
    prompt_version: str | None = "assistant@v1",
) -> ReplayRequest:
    body: dict = {
        "model": "claude-opus-5",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question or f"question number {n} about caching",
                    }
                ],
            }
        ],
    }
    if system is not None:
        body["system"] = [{"type": "text", "text": system}]
    return ReplayRequest(
        request_id=f"req-{n}",
        tenant_id=tenant,
        timestamp=at or datetime(2026, 8, 23, 12, 0, n % 60, tzinfo=UTC),
        body=body,
        response={"content": [{"type": "text", "text": f"answer {n}"}]},
        usage=TokenUsage(input_tokens=4000, output_tokens=200),
        prompt_version=prompt_version,
    )


async def run(corpus, prompts_root, *, threshold: float = 0.99, **kwargs):
    return await replay(
        corpus,
        HashingEmbedder(),
        threshold=threshold,
        prompts_root=prompts_root,
        **kwargs,
    )


# --- the shape of the comparison ------------------------------------------


async def test_all_three_configurations_see_the_same_corpus(prompts):
    report = await run([request(i) for i in range(10)], prompts)
    assert report.no_cache.requests == 10
    assert report.prefix_only.requests == 10
    assert report.prefix_and_semantic.requests == 10


async def test_no_cache_pays_full_price_for_every_input_token(prompts):
    report = await run([request(i) for i in range(5)], prompts)
    config = report.no_cache
    assert config.upstream_calls == 5
    assert config.usage.cache_read_input_tokens == 0
    assert config.usage.cache_creation_input_tokens == 0
    assert config.usage.input_tokens == 5 * 4000


async def test_a_repeated_prefix_becomes_a_read_after_the_first_write(prompts):
    # The prompt is registry-versioned and identical across requests, so it is
    # written once and read thereafter.
    report = await run([request(i) for i in range(6)], prompts)
    config = report.prefix_only
    assert config.prefix_writes == 1
    assert config.prefix_reads == 5
    assert config.usage.cache_read_input_tokens > 0


async def test_prefix_caching_is_cheaper_than_none_on_a_repeating_corpus(prompts):
    report = await run([request(i) for i in range(20)], prompts)
    assert report.prefix_only.cost < report.no_cache.cost
    assert report.prefix_saving > 0


async def test_an_unversioned_prompt_gets_no_prefix_caching_at_all(prompts):
    # The safety default costs the optimisation, which is the right way round.
    corpus = [request(i, prompt_version=None) for i in range(6)]
    report = await run(corpus, prompts)
    assert report.prefix_only.prefix_writes == 0
    assert report.prefix_only.prefix_reads == 0
    assert report.prefix_only.cost == report.no_cache.cost


async def test_a_prefix_that_expires_is_written_again(prompts):
    # Beyond the TTL window the entry is gone, so the next sighting is a write.
    corpus = [
        request(0, at=datetime(2026, 8, 23, 12, 0, 0, tzinfo=UTC)),
        request(1, at=datetime(2026, 8, 23, 12, 0, 30, tzinfo=UTC)),
        request(2, at=datetime(2026, 8, 23, 12, 30, 0, tzinfo=UTC)),
    ]
    report = await run(corpus, prompts, ttl="5m")
    assert report.prefix_only.prefix_writes == 2
    assert report.prefix_only.prefix_reads == 1


async def test_tenants_do_not_share_a_simulated_prefix(prompts):
    # Two tenants, same prompt: different scope keys, so each pays its own write.
    corpus = [request(0, tenant="a"), request(1, tenant="b")]
    report = await run(corpus, prompts)
    assert report.prefix_only.prefix_writes == 2
    assert report.prefix_only.prefix_reads == 0


# --- the incremental number -----------------------------------------------


async def test_a_semantic_hit_removes_the_upstream_call_entirely(prompts):
    question = "How do I reverse a list in Python, step by step, with an example?"
    corpus = [request(0, question=question), request(1, question=question)]
    report = await run(corpus, prompts, threshold=0.5)

    assert report.prefix_and_semantic.semantic_hits == 1
    # One fewer call than the prefix-only arm, which had to make both.
    assert report.prefix_and_semantic.upstream_calls == 1
    assert report.prefix_only.upstream_calls == 2


async def test_the_incremental_saving_is_measured_against_prefix_not_against_none(
    prompts,
):
    question = "How do I reverse a list in Python, step by step, with an example?"
    corpus = [request(i, question=question) for i in range(10)]
    report = await run(corpus, prompts, threshold=0.5)

    # All three numbers exist and mean different things. The incremental one is
    # smaller than the total, because prefix caching already took the easy wins
    # at no correctness risk.
    assert report.prefix_saving > 0
    assert report.incremental_saving > 0
    assert report.total_saving > report.incremental_saving


async def test_a_corpus_with_no_repeats_gets_nothing_from_the_semantic_layer(prompts):
    # The honest negative result: distinct questions, no hits, no incremental
    # saving, and the exposure would have been added for nothing.
    questions = [
        "What is the boiling point of water at sea level in Celsius?",
        "Explain how a suspension bridge distributes load across its towers.",
        "Who composed the opera Der Rosenkavalier and in which year?",
        "Describe the lifecycle of a monarch butterfly in four stages.",
        "What distinguishes sedimentary rock from igneous rock geologically?",
        "How does a heat pump move thermal energy against a gradient?",
        "Summarise the plot of Dostoevsky's novel The Idiot briefly.",
        "Which enzymes are responsible for breaking down starch in saliva?",
    ]
    corpus = [request(i, question=q) for i, q in enumerate(questions)]
    report = await run(corpus, prompts, threshold=0.99)
    assert report.prefix_and_semantic.semantic_hits == 0
    assert report.incremental_saving == pytest.approx(0.0, abs=1e-9)
    # ...while prefix caching still pays, because the prompt repeats.
    assert report.prefix_saving > 0


async def test_the_report_says_it_is_a_simulation(prompts):
    # A number produced by modelling the provider's cache state is not a live
    # measurement, and any report built on it has to carry that.
    report = await run([request(i) for i in range(3)], prompts)
    assert report.as_json()["simulation"] is True
    assert "SIMULATED" in report.render()


async def test_a_small_increment_is_called_out_rather_than_dressed_up(prompts):
    report = await run([request(i) for i in range(3)], prompts, threshold=0.999999)
    report.false_hit_rate = 0.0
    report.false_hit_upper_bound = 0.12
    rendered = report.render()
    assert "stronger result than a headline hit rate" in rendered


# --- auditing the hits ----------------------------------------------------


async def test_every_semantic_hit_can_be_inspected(prompts):
    # A hit rate is only trustworthy if the hits can be read. This is what turns
    # the exposure figure into a measured number rather than an assumed one.
    question = "How do I reverse a list in Python, step by step, with an example?"
    corpus = [request(0, question=question), request(1, question=question)]
    report = await run(corpus, prompts, threshold=0.5)

    audit = audit_served(report, corpus)
    assert len(audit) == 1
    row = audit[0]
    assert row["request_id"] == "req-1"
    assert row["asked"] == question
    assert row["served_answer_to"] == question
    assert row["similarity"] >= 0.5
    # Left blank on purpose: a human decides, the tool does not guess.
    assert row["false_hit"] is None
