"""Layer 2: the semantic cache.

Unlike the prefix cache, this one can be **wrong**. A miss costs money; a false
hit returns someone else's answer to a real user. Those are not comparable
failures, and every decision here follows from that asymmetry:

* The threshold is calibrated against hand-labelled pairs with precision weighted
  far above recall, not tuned until the hit rate looks good.
* The cache refuses to run on a non-production embedder, because "nearest" under
  a hashing stub means nothing.
* Scope is a partition, not a filter (see `keys.py`).
* Streaming responses are only cached when the stream completed — a truncated
  answer is a wrong cache entry, not a cheap one.

**The number worth reporting is the incremental one.** Prefix caching already
captures the easy wins at zero correctness risk, so this layer must justify
itself on the residual: the cost delta of `prefix + semantic` against `prefix
alone`, at a stated false-hit rate. If that margin turns out to be small at a
defensible operating point, reporting it honestly is a better result than a
headline hit rate that hides the exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embeddings import Embedder
from .keys import CacheScope, query_text
from .store import CacheEntry, SemanticCacheStore


class UnsafeEmbedderError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheHit:
    response: dict[str, Any]
    similarity: float
    matched_query: str
    entry_id: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "hit": True,
            "similarity": self.similarity,
            "matched_query": self.matched_query[:200],
            "entry_id": self.entry_id,
        }


@dataclass(frozen=True)
class CacheMiss:
    """A miss, and how close it came.

    `best_similarity` is recorded even on a miss. It is what the threshold sweep
    in `calibration.py` runs on, so production traffic becomes the data for
    re-tuning the operating point rather than requiring a separate labelling
    exercise.
    """

    best_similarity: float | None = None
    nearest_query: str | None = None
    reason: str = "no neighbour above threshold"

    def as_json(self) -> dict[str, Any]:
        return {
            "hit": False,
            "best_similarity": self.best_similarity,
            "nearest_query": (self.nearest_query or "")[:200] or None,
            "reason": self.reason,
        }


@dataclass
class SemanticCacheConfig:
    enabled: bool = False  # off until a threshold has been calibrated
    threshold: float = 0.95
    # Refuse to serve when the query is very short. A three-word query has few
    # tokens to disambiguate it, so cosine similarity between two different
    # short queries runs high and false hits cluster here.
    min_query_chars: int = 24
    # Only cache responses that cost more than this. Caching a cheap response
    # spends storage and false-hit risk to save almost nothing.
    min_cost_usd: float = 0.0


class SemanticCache:
    def __init__(
        self,
        store: SemanticCacheStore,
        embedder: Embedder,
        config: SemanticCacheConfig | None = None,
        *,
        allow_unsafe_embedder: bool = False,
    ) -> None:
        if not embedder.is_production_grade and not allow_unsafe_embedder:
            raise UnsafeEmbedderError(
                f"embedder {embedder.model_name!r} is not production grade. Nearest "
                "neighbours under it are not semantically near, so a hit would be "
                "an arbitrary wrong answer. Pass allow_unsafe_embedder=True only "
                "in tests."
            )
        self.store = store
        self.embedder = embedder
        self.config = config or SemanticCacheConfig()

    async def lookup(self, body: dict[str, Any], scope: CacheScope) -> CacheHit | CacheMiss:
        if not self.config.enabled:
            return CacheMiss(reason="semantic cache disabled")

        query = query_text(body)
        if len(query) < self.config.min_query_chars:
            return CacheMiss(
                reason=f"query shorter than {self.config.min_query_chars} chars; "
                "short queries are where false hits cluster"
            )

        embedding = self.embedder.encode([query], is_query=True)[0]
        neighbours = await self.store.nearest(scope, embedding, limit=1)
        if not neighbours:
            return CacheMiss(reason="scope is empty")

        best = neighbours[0]
        if best.similarity < self.config.threshold:
            return CacheMiss(
                best_similarity=best.similarity,
                nearest_query=best.entry.query_text,
                reason=f"nearest {best.similarity:.4f} < threshold {self.config.threshold:.4f}",
            )

        return CacheHit(
            response=best.entry.response,
            similarity=best.similarity,
            matched_query=best.entry.query_text,
            entry_id=best.entry.entry_id,
        )

    async def put(
        self,
        body: dict[str, Any],
        scope: CacheScope,
        response: dict[str, Any],
        *,
        cacheable: bool = True,
        cost_usd: float = 0.0,
    ) -> bool:
        """Store a completion. Returns whether it was actually stored."""
        if not self.config.enabled:
            return False
        if not cacheable:
            # A truncated stream, or a response the caller flagged for any other
            # reason. Storing a partial answer is a wrong cache entry.
            return False
        if cost_usd < self.config.min_cost_usd:
            return False

        query = query_text(body)
        if len(query) < self.config.min_query_chars:
            return False

        # Stored without the query prefix: entries are the "passage" side of the
        # asymmetric bge model, and prefixing both sides degrades retrieval.
        embedding = self.embedder.encode([query], is_query=False)[0]
        await self.store.put(
            CacheEntry(
                scope_key=scope.key,
                query_text=query,
                embedding=embedding,
                response=response,
            )
        )
        return True
