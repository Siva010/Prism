"""Tests that need a real Postgres, skipped when there isn't one.

Everything else in this suite runs against fakes, which is right — a unit suite
that needs a database is a unit suite nobody runs. But the gap that leaves is
real: the migration runner shipped broken for six weeks because migration 0001
was applied by the Postgres container's initdb entrypoint and the runner itself
was never exercised. asyncpg rejects multi-statement SQL through a prepared
statement, and every path through SQLAlchemy hands it one.

So the rule is: anything whose failure mode only appears against a real server
gets a test here.

    docker compose up -d && python scripts/migrate.py && pytest tests/test_integration_db.py
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid

import pytest

from prism.caching.embeddings import HashingEmbedder, cosine
from prism.caching.keys import CacheScope
from prism.caching.store import CacheEntry, InMemoryStore, PgVectorStore, purge_scope
from prism.db.engine import dispose_engine, get_sessionmaker, init_engine, session_scope

DATABASE_URL = os.environ.get(
    "PRISM_TEST_DATABASE_URL",
    "postgresql+asyncpg://prism:prism@localhost:5434/prism",
)


def _reachable() -> bool:
    async def probe() -> bool:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(DATABASE_URL)
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False
        finally:
            await engine.dispose()

    with contextlib.suppress(Exception):
        return asyncio.run(probe())
    return False


pytestmark = pytest.mark.skipif(
    not _reachable(),
    reason=f"no Postgres at {DATABASE_URL}; run `docker compose up -d`",
)


@pytest.fixture
async def engine():
    init_engine(DATABASE_URL)
    yield
    await dispose_engine()


def body(text: str) -> dict:
    return {
        "model": "claude-opus-5",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }


async def test_every_migration_has_been_applied(engine):
    from sqlalchemy import text

    async with session_scope() as session:
        applied = {
            row[0] for row in (await session.execute(text("SELECT version FROM schema_migrations")))
        }
    assert {"0001_init", "0002_eval", "0003_semantic_cache"} <= applied


async def test_the_expected_tables_exist(engine):
    from sqlalchemy import text

    async with session_scope() as session:
        tables = {
            row[0]
            for row in (
                await session.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                )
            )
        }
    assert {
        "tenants",
        "api_keys",
        "traces",
        "eval_runs",
        "eval_results",
        "human_labels",
        "semantic_cache",
        "cache_calibrations",
    } <= tables


async def test_the_hnsw_index_exists(engine):
    # A plain btree over a vector column would silently give correct results and
    # terrible latency, so the index *type* is what matters here.
    from sqlalchemy import text

    async with session_scope() as session:
        definition = (
            await session.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE indexname = 'semantic_cache_embedding_idx'"
                )
            )
        ).scalar_one()
    assert "USING hnsw" in definition
    assert "vector_cosine_ops" in definition


async def test_pgvector_distance_is_converted_to_cosine_similarity(engine):
    # pgvector's <=> is cosine *distance*. Getting the conversion backwards would
    # invert every ranking while still returning plausible-looking numbers.
    store = PgVectorStore(get_sessionmaker())
    embedder = HashingEmbedder(dimension=384)
    scope = CacheScope.from_request(body("q"), tenant_id=f"itest-{uuid.uuid4().hex[:8]}")

    stored = "How do I reverse a list in Python, with a worked example please?"
    probe = "How do I reverse a list in Python, with a worked example?"
    vector = embedder.encode([stored])[0]
    await store.put(CacheEntry(scope.key, stored, vector, {"a": 1}))

    try:
        found = await store.nearest(scope, embedder.encode([probe])[0], limit=1)
        assert found
        expected = cosine(embedder.encode([probe])[0], vector)
        assert found[0].similarity == pytest.approx(expected, abs=1e-3)
    finally:
        async with session_scope() as session:
            await purge_scope(session, scope.key)


async def test_the_two_stores_agree(engine):
    # The week-8 replay numbers were measured on InMemoryStore. They only carry
    # over to production if pgvector returns the same similarities.
    embedder = HashingEmbedder(dimension=384)
    scope = CacheScope.from_request(body("q"), tenant_id=f"itest-{uuid.uuid4().hex[:8]}")
    stored, probe = (
        "a reasonably long stored question about caching",
        "a reasonably long stored question on caching",
    )
    vector = embedder.encode([stored])[0]

    pg = PgVectorStore(get_sessionmaker())
    memory = InMemoryStore()
    await pg.put(CacheEntry(scope.key, stored, vector, {"a": 1}))
    await memory.put(CacheEntry(scope.key, stored, vector, {"a": 1}))

    try:
        query = embedder.encode([probe])[0]
        from_pg = (await pg.nearest(scope, query, limit=1))[0].similarity
        from_memory = (await memory.nearest(scope, query, limit=1))[0].similarity
        assert from_pg == pytest.approx(from_memory, abs=1e-3)
    finally:
        async with session_scope() as session:
            await purge_scope(session, scope.key)


async def test_scope_isolation_holds_at_the_sql_level(engine):
    # The partition must be in the WHERE clause, not a Python filter after
    # retrieval — otherwise the leak exists in the query path.
    store = PgVectorStore(get_sessionmaker())
    embedder = HashingEmbedder(dimension=384)
    suffix = uuid.uuid4().hex[:8]
    question = "What is our internal audit log retention period, exactly?"

    mine = CacheScope.from_request(body(question), tenant_id=f"itest-a-{suffix}")
    theirs = CacheScope.from_request(body(question), tenant_id=f"itest-b-{suffix}")
    vector = embedder.encode([question])[0]
    await store.put(CacheEntry(mine.key, question, vector, {"secret": "31 days"}))

    try:
        assert await store.nearest(mine, vector, limit=1)
        # Identical query, similarity 1.0, and it must still return nothing.
        assert await store.nearest(theirs, vector, limit=1) == []
    finally:
        async with session_scope() as session:
            await purge_scope(session, mine.key)


async def test_purging_a_scope_reclaims_its_entries(engine):
    # Promoting a prompt version changes the scope key, so old entries become
    # unreachable but still occupy the index.
    store = PgVectorStore(get_sessionmaker())
    embedder = HashingEmbedder(dimension=384)
    scope = CacheScope.from_request(body("q"), tenant_id=f"itest-{uuid.uuid4().hex[:8]}")
    for i in range(3):
        text_i = f"a stored question number {i} about caching behaviour"
        await store.put(CacheEntry(scope.key, text_i, embedder.encode([text_i])[0], {"i": i}))

    async with session_scope() as session:
        assert await purge_scope(session, scope.key) == 3
    assert await store.nearest(scope, embedder.encode(["anything"])[0], limit=1) == []


# --- governance (week 10-11) ----------------------------------------------


async def _tenant(session) -> uuid.UUID:
    from sqlalchemy import text

    tid = uuid.uuid4()
    await session.execute(
        text("INSERT INTO tenants (id, slug, name) VALUES (:i, :s, :n)"),
        {"i": str(tid), "s": f"budget-{tid.hex[:8]}", "n": "Budget Test"},
    )
    await session.commit()
    return tid


async def test_a_tenant_without_a_budget_is_unlimited(engine):
    from prism.governance import budgets

    async with session_scope() as session:
        tid = await _tenant(session)
        decision = await budgets.check(session, tid)
    assert decision.allowed
    assert decision.status is budgets.BudgetStatus.OK
    assert decision.hard_cap_usd is None


async def test_the_soft_cap_warns_and_keeps_serving(engine):
    # The whole reason there are two caps: one alone forces a choice between
    # surprising people with a bill and cutting them off without warning.
    from decimal import Decimal

    from prism.governance import budgets

    async with session_scope() as session:
        tid = await _tenant(session)
        await budgets.set_budget(
            session, tid, soft_cap_usd=Decimal("1.00"), hard_cap_usd=Decimal("10.00")
        )
        await budgets.record_spend(session, tid, Decimal("2.50"))
        decision = await budgets.check(session, tid)

    assert decision.status is budgets.BudgetStatus.SOFT_CAP
    assert decision.allowed  # still serving
    assert not decision.must_degrade
    assert decision.utilisation == pytest.approx(0.25)


async def test_the_hard_cap_can_reject_or_degrade(engine):
    from decimal import Decimal

    from prism.governance import budgets

    async with session_scope() as session:
        rejecting = await _tenant(session)
        await budgets.set_budget(
            session, rejecting, hard_cap_usd=Decimal("1.00"), hard_cap_action="reject"
        )
        await budgets.record_spend(session, rejecting, Decimal("1.50"))
        decision = await budgets.check(session, rejecting)
        assert decision.status is budgets.BudgetStatus.HARD_CAP_REJECT
        assert not decision.allowed

        # Degrading usually serves a tenant better than a 402: they keep working
        # at a fraction of the cost.
        degrading = await _tenant(session)
        await budgets.set_budget(
            session, degrading, hard_cap_usd=Decimal("1.00"), hard_cap_action="degrade"
        )
        await budgets.record_spend(session, degrading, Decimal("1.50"))
        decision = await budgets.check(session, degrading)
        assert decision.must_degrade
        assert decision.allowed


async def test_a_soft_cap_above_the_hard_cap_is_refused(engine):
    # It could never fire, so the tenant would be cut off having never been warned.
    from decimal import Decimal

    from prism.governance import budgets

    async with session_scope() as session:
        tid = await _tenant(session)
        with pytest.raises(ValueError, match="would never fire"):
            await budgets.set_budget(
                session, tid, soft_cap_usd=Decimal("50"), hard_cap_usd=Decimal("10")
            )


async def test_spend_resets_when_the_period_rolls_over(engine):
    from datetime import date
    from decimal import Decimal

    from prism.governance import budgets

    async with session_scope() as session:
        tid = await _tenant(session)
        await budgets.set_budget(session, tid, hard_cap_usd=Decimal("1.00"), period="month")
        await budgets.record_spend(session, tid, Decimal("5.00"))
        assert not (await budgets.check(session, tid)).allowed

        # Next month: a stale total would keep them capped into a period they
        # have not spent anything in.
        later = date(2099, 1, 15)
        decision = await budgets.check(session, tid, today=later)
    assert decision.allowed
    assert decision.spent_usd == 0


async def test_reconcile_rebuilds_the_total_from_traces(engine):
    # The running total is a cache; traces are the truth. A process that dies
    # between the upstream call and the update loses accuracy, not the ledger.
    from decimal import Decimal

    from sqlalchemy import text

    from prism.governance import budgets

    async with session_scope() as session:
        tid = await _tenant(session)
        await budgets.set_budget(session, tid, hard_cap_usd=Decimal("100"))
        for cost in ("0.25", "0.50"):
            await session.execute(
                text("""
                    INSERT INTO traces (tenant_id, endpoint, model_requested,
                                        model_resolved, status, cost_usd)
                    VALUES (:t, '/v1/chat/completions', 'm', 'm', 'ok', :c)
                """),
                {"t": str(tid), "c": cost},
            )
        await session.commit()

        # The denormalized figure is stale...
        assert (await budgets.check(session, tid)).spent_usd == 0
        rebuilt = await budgets.reconcile(session, tid)

    assert rebuilt == Decimal("0.75")


async def test_the_cost_attribution_view_divides_by_successful_tasks(engine):
    # Cost per SUCCESSFUL task, not per request. A tier that fails 40% of the
    # time and retries elsewhere is not cheap, and only this denominator shows it.
    from sqlalchemy import text

    async with session_scope() as session:
        tid = await _tenant(session)
        rows = [("ok", "0.10"), ("ok", "0.10"), ("error", "0.05")]
        for status, cost in rows:
            await session.execute(
                text("""
                    INSERT INTO traces (tenant_id, endpoint, model_requested,
                                        model_resolved, status, cost_usd)
                    VALUES (:t, '/v1/chat/completions', 'm', 'claude-opus-5', :s, :c)
                """),
                {"t": str(tid), "s": status, "c": cost},
            )
        await session.commit()

        row = (
            (
                await session.execute(
                    text("""
                    SELECT requests, successful, cost_usd, cost_per_successful_task
                    FROM cost_attribution WHERE tenant_id = :t
                """),
                    {"t": str(tid)},
                )
            )
            .mappings()
            .first()
        )

    assert row["requests"] == 3
    assert row["successful"] == 2
    assert float(row["cost_usd"]) == pytest.approx(0.25)
    # 0.25 over 2, not over 3 — the failed request's spend still counts.
    assert float(row["cost_per_successful_task"]) == pytest.approx(0.125)


async def test_a_budget_event_records_why_a_tenant_was_cut_off(engine):
    from decimal import Decimal

    from sqlalchemy import text

    from prism.governance import budgets

    async with session_scope() as session:
        tid = await _tenant(session)
        await budgets.set_budget(session, tid, hard_cap_usd=Decimal("1.00"))
        await budgets.record_spend(session, tid, Decimal("1.50"))
        decision = await budgets.check(session, tid)
        await budgets.record_event(session, tid, "hard_cap", decision, {"note": "test"})

        event = (
            (
                await session.execute(
                    text("SELECT event, spent_usd, detail FROM budget_events WHERE tenant_id = :t"),
                    {"t": str(tid)},
                )
            )
            .mappings()
            .first()
        )

    assert event["event"] == "hard_cap"
    assert float(event["spent_usd"]) == pytest.approx(1.5)
