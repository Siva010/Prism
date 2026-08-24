"""Embeddings for the semantic cache.

**Why the model runs locally.** Calling a paid embedding API to decide whether
you can avoid a paid completion API defeats the entire exercise. At roughly
$0.02/M tokens for a hosted embedding against $5/M for Opus input, a hosted
embedder still eats a visible slice of the saving it exists to produce — and it
adds a network round trip to the hot path of every request, cached or not.
`bge-small-en-v1.5` is ~33M parameters, runs in single-digit milliseconds on CPU,
and scores well on MTEB for its size. The saving is then bounded only by the
electricity.

Two implementations, and the distinction between them is enforced rather than
documented: `LocalEmbedder` is the real one, and `HashingEmbedder` is a
deterministic stand-in for tests that reports `is_production_grade = False`. The
cache refuses to serve traffic on a non-production embedder, because a hash-based
"similarity" would produce nearest neighbours that are not semantically near at
all — which in this layer means returning a wrong answer to a real user.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSION = 384

# bge models are trained with an instruction prefix on the *query* side only.
# Omitting it costs a few points of retrieval quality, which here translates
# directly into a worse precision/recall curve.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str
    dimension: int
    is_production_grade: bool

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Both embedders normalize, so this is a dot product."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return max(-1.0, min(1.0, dot))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class EmbeddingUnavailableError(RuntimeError):
    pass


class LocalEmbedder:
    """`bge-small-en-v1.5` on CPU via sentence-transformers.

    The model is loaded lazily on first use: importing torch costs seconds and
    hundreds of megabytes, and a gateway with the semantic cache disabled should
    not pay either.
    """

    is_production_grade = True

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cpu",
        cache_size: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.dimension = DEFAULT_DIMENSION
        self._device = device
        self._model: Any = None
        # In-process LRU over hot query strings. Repeated identical requests are
        # common, and re-embedding one costs more than the dictionary lookup that
        # avoids it.
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size
        self.hits = 0
        self.misses = 0

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingUnavailableError(
                    "sentence-transformers is not installed. Install the extra:\n"
                    '    uv pip install -e ".[embeddings]"\n'
                    "The semantic cache cannot run without a real embedding model."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self._device)
            # Renamed in sentence-transformers 6; support both so a version
            # bump does not turn into a runtime warning on every load.
            getter = getattr(
                self._model,
                "get_embedding_dimension",
                getattr(self._model, "get_sentence_embedding_dimension", None),
            )
            if getter is not None:
                self.dimension = int(getter())
        return self._model

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        prepared = [QUERY_PREFIX + t if is_query else t for t in texts]

        out: list[list[float] | None] = [None] * len(prepared)
        to_encode: list[str] = []
        positions: list[int] = []
        for i, text in enumerate(prepared):
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                out[i] = cached
                self.hits += 1
            else:
                to_encode.append(text)
                positions.append(i)
                self.misses += 1

        if to_encode:
            model = self._load()
            vectors = model.encode(to_encode, normalize_embeddings=True, show_progress_bar=False)
            for position, vector in zip(positions, vectors, strict=True):
                value = [float(x) for x in vector]
                out[position] = value
                self._cache[prepared[position]] = value
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

        return [v for v in out if v is not None]


class HashingEmbedder:
    """Deterministic stand-in. **Not** semantically meaningful.

    Exists so the cache's plumbing — key scoping, thresholds, isolation, the
    store — can be tested without a 400MB model download in CI. Similarity
    between two of these vectors reflects nothing but hash collisions, so
    `is_production_grade` is False and the cache refuses to serve traffic on it.
    """

    is_production_grade = False

    def __init__(self, dimension: int = 64) -> None:
        self.model_name = "hashing-stub"
        self.dimension = dimension

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        # Character trigrams give near-identical strings near-identical vectors,
        # which is enough to exercise a threshold in a test.
        tokens = [text[i : i + 3] for i in range(max(1, len(text) - 2))]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = struct.unpack("<I", digest[:4])[0] % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)


def build_embedder(kind: str, **kwargs: object) -> Embedder:
    if kind == "local":
        return LocalEmbedder(**kwargs)  # type: ignore[arg-type]
    if kind == "hashing":
        return HashingEmbedder(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown embedder {kind!r}; expected 'local' or 'hashing'")
