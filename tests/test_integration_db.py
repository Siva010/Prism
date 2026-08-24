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
