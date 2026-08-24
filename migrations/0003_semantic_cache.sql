-- Prism 0003 - semantic cache (layer 2) and its calibration record.
-- Idempotent: safe to re-run.

-- 384 dims = bge-small-en-v1.5. Changing the embedding model changes this, and
-- entries written under a different model are not comparable, so a model change
-- means dropping the table rather than migrating it.
CREATE TABLE IF NOT EXISTS semantic_cache (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_hit_at  TIMESTAMPTZ,
    hits         INTEGER NOT NULL DEFAULT 0,

    -- The partition. Tenant, model, temperature, system-prompt hash and
    -- tool-schema hash are folded into this one value, and a nearest-neighbour
    -- search never crosses it. Filtering after retrieval would mean the leak
    -- exists in the query path and is prevented only by a later condition.
    scope_key    TEXT NOT NULL,

    query_text   TEXT NOT NULL,
    embedding    vector(384) NOT NULL,
    response     JSONB NOT NULL,

    cost_usd     NUMERIC(14, 8) NOT NULL DEFAULT 0,
    extra        JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- The scope filter must be selective before the vector scan runs, so it is a
-- plain btree index rather than part of the HNSW index.
CREATE INDEX IF NOT EXISTS semantic_cache_scope_idx ON semantic_cache (scope_key);

-- HNSW over cosine distance. m and ef_construction are build-time
-- recall/latency dials; ef_search is set per session (see store.PgVectorStore)
-- because it is the one worth tuning after measuring real recall.
CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx
    ON semantic_cache USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS semantic_cache_created_idx ON semantic_cache (created_at DESC);

-- Every calibration run, kept so a threshold in production can always be traced
-- back to the labelled set and embedder that justified it.
CREATE TABLE IF NOT EXISTS cache_calibrations (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedder             TEXT NOT NULL,
    n_pairs              INTEGER NOT NULL,
    auc                  DOUBLE PRECISION,
    max_false_hit_rate   DOUBLE PRECISION NOT NULL,
    chosen_threshold     DOUBLE PRECISION,
    -- Null when no threshold met the ceiling. That is a real result: it means
    -- the cache should stay off for this corpus.
    chosen               JSONB,
    curve                JSONB NOT NULL DEFAULT '[]'::jsonb,
    git_sha              TEXT
);

CREATE INDEX IF NOT EXISTS cache_calibrations_created_idx
    ON cache_calibrations (created_at DESC);

INSERT INTO schema_migrations (version) VALUES ('0003_semantic_cache')
ON CONFLICT (version) DO NOTHING;
