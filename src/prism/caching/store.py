"""Where cached completions live.

Two implementations behind one protocol. `PgVectorStore` is production: vectors
and traces in the same database, which is the reason for choosing pgvector over a
standalone vector store — cache-hit data can be joined against cost and quality
data in a single query. `InMemoryStore` is for tests and for the offline replay
harness, which compares cache configurations over a fixed corpus and has no
reason to touch a database.

Both partition by scope key before comparing anything. See `keys.py` for why that
is partitioning rather than post-filtering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .embeddings import cosine
from .keys import CacheScope


@dataclass(frozen=True)
class CacheEntry:
    scope_key: str
    query_text: str
    embedding: list[float]
    response: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    entry_id: str | None = None


@dataclass(frozen=True)
class Neighbour:
    entry: CacheEntry
    similarity: float


class SemanticCacheStore(Protocol):
    async def nearest(
        self, scope: CacheScope, embedding: list[float], *, limit: int = 1
    ) -> list[Neighbour]: ...

    async def put(self, entry: CacheEntry) -> None: ...

    async def size(self) -> int: ...


class InMemoryStore:
    """Exhaustive scan within a scope. Exact, and fine for tests and replay.

    Deliberately not approximate: the replay harness measures how a *threshold*
    behaves, and mixing in ANN recall error would confound the calibration with
    the index's own misses.
    """

    def __init__(self) -> None:
        self._by_scope: dict[str, list[CacheEntry]] = {}

    async def nearest(
        self, scope: CacheScope, embedding: list[float], *, limit: int = 1
    ) -> list[Neighbour]:
        candidates = self._by_scope.get(scope.key, [])
        scored = [Neighbour(e, cosine(embedding, e.embedding)) for e in candidates]
        scored.sort(key=lambda n: n.similarity, reverse=True)
        return scored[:limit]

    async def put(self, entry: CacheEntry) -> None:
        self._by_scope.setdefault(entry.scope_key, []).append(entry)

    async def size(self) -> int:
        return sum(len(v) for v in self._by_scope.values())

    def clear(self) -> None:
        self._by_scope.clear()


class PgVectorStore:
    """pgvector + HNSW.

    `m`, `ef_construction`, and `ef_search` are the recall-versus-latency dials.
    `ef_search` is set per session rather than baked into the index because it is
    the one that can be tuned against measured recall after the fact — raising it
    costs latency and buys recall, and which side of that trade is right depends
    on the false-hit exposure the threshold is already accepting.
    """

    def __init__(self, session_factory: Any, *, ef_search: int = 40) -> None:
        self._session_factory = session_factory
        self.ef_search = ef_search

    async def nearest(
        self, scope: CacheScope, embedding: list[float], *, limit: int = 1
    ) -> list[Neighbour]:
        vector = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
        async with self._session_factory() as session:
            await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}"))
            rows = (
                await session.execute(
                    text("""
                        SELECT id, query_text, response, embedding <=> CAST(:v AS vector)
                               AS distance
                        FROM semantic_cache
                        WHERE scope_key = :scope
                        ORDER BY embedding <=> CAST(:v AS vector)
                        LIMIT :limit
                    """),
                    {"v": vector, "scope": scope.key, "limit": limit},
                )
            ).all()

        out: list[Neighbour] = []
        for row in rows:
            response = row.response
            if isinstance(response, str):
                response = json.loads(response)
            out.append(
                Neighbour(
                    entry=CacheEntry(
                        scope_key=scope.key,
                        query_text=row.query_text,
                        embedding=[],
                        response=response,
                        entry_id=str(row.id),
                    ),
                    # pgvector's <=> is cosine *distance*; similarity is 1 - it.
                    similarity=1.0 - float(row.distance),
                )
            )
        return out

    async def put(self, entry: CacheEntry) -> None:
        vector = "[" + ",".join(f"{x:.6f}" for x in entry.embedding) + "]"
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO semantic_cache
                        (scope_key, query_text, embedding, response)
                    VALUES (:scope, :q, CAST(:v AS vector), CAST(:r AS JSONB))
                """),
                {
                    "scope": entry.scope_key,
                    "q": entry.query_text,
                    "v": vector,
                    "r": json.dumps(entry.response),
                },
            )
            await session.commit()

    async def size(self) -> int:
        async with self._session_factory() as session:
            return int(
                (await session.execute(text("SELECT count(*) FROM semantic_cache"))).scalar_one()
            )


async def purge_scope(session: AsyncSession, scope_key: str) -> int:
    """Drop every entry in one scope.

    The invalidation primitive. Promoting a prompt version changes the scope key,
    so old entries become unreachable rather than stale — but they still occupy
    the index, and this is what reclaims them.
    """
    result = await session.execute(
        text("DELETE FROM semantic_cache WHERE scope_key = :scope"), {"scope": scope_key}
    )
    await session.commit()
    return int(getattr(result, "rowcount", 0) or 0)
