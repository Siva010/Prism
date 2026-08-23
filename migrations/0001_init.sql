-- Prism 0001 — tenants, API keys, traces.
-- Idempotent: safe to re-run. Applied automatically on first `docker compose up`
-- (mounted into /docker-entrypoint-initdb.d) and by `python scripts/migrate.py`.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
-- Not used until the semantic cache (weeks 7-8), but enabling it here keeps the
-- vector store and the trace store in the same database from day one, which is
-- the whole point of choosing pgvector over a standalone vector DB.
CREATE EXTENSION IF NOT EXISTS "vector";

CREATE TABLE IF NOT EXISTS tenants (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active   BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    -- SHA-256 of the presented key. The plaintext key is shown once, at creation.
    key_hash    TEXT NOT NULL UNIQUE,
    -- Non-secret prefix so a key is identifiable in the dashboard without storing it.
    key_prefix  TEXT NOT NULL,
    label       TEXT NOT NULL DEFAULT '',
    scopes      TEXT[] NOT NULL DEFAULT ARRAY['chat']::TEXT[],
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    revoked_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS api_keys_tenant_idx ON api_keys (tenant_id);

CREATE TABLE IF NOT EXISTS traces (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    tenant_id             UUID REFERENCES tenants(id) ON DELETE SET NULL,
    api_key_id            UUID REFERENCES api_keys(id) ON DELETE SET NULL,

    endpoint              TEXT NOT NULL,
    ingress_format        TEXT NOT NULL DEFAULT 'openai',
    provider              TEXT NOT NULL DEFAULT 'anthropic',
    model_requested       TEXT NOT NULL,
    model_resolved        TEXT NOT NULL,
    stream                BOOLEAN NOT NULL DEFAULT FALSE,

    status                TEXT NOT NULL,              -- ok | error
    http_status           INTEGER,
    error_type            TEXT,                       -- 429 vs 529 vs timeout vs translation
    error_message         TEXT,

    stop_reason           TEXT,                       -- Anthropic vocabulary
    finish_reason         TEXT,                       -- OpenAI vocabulary, as returned

    -- Latency is split because a single number cannot express "fast first token,
    -- slow total" vs "slow first token". ttft_ms is populated for streaming only.
    latency_ms            INTEGER,
    ttft_ms               INTEGER,
    upstream_latency_ms   INTEGER,

    -- Five token classes, priced differently. Never collapse these into a total.
    input_tokens              INTEGER NOT NULL DEFAULT 0,  -- uncached input
    cache_read_input_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens             INTEGER NOT NULL DEFAULT 0,  -- includes thinking tokens
    thinking_tokens           INTEGER NOT NULL DEFAULT 0,  -- subset of output, when reported

    cost_usd              NUMERIC(14, 8) NOT NULL DEFAULT 0,
    cost_breakdown        JSONB NOT NULL DEFAULT '{}'::jsonb,

    prompt_version        TEXT,
    provider_request_id   TEXT,

    -- Bodies are nullable so PRISM_TRACE_BODIES=false yields metrics-only traces.
    request_body          JSONB,
    upstream_request      JSONB,
    upstream_response     JSONB,
    response_body         JSONB,

    extra                 JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS traces_tenant_created_idx ON traces (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS traces_created_idx        ON traces (created_at DESC);
CREATE INDEX IF NOT EXISTS traces_model_idx          ON traces (model_resolved, created_at DESC);
CREATE INDEX IF NOT EXISTS traces_status_idx         ON traces (status) WHERE status <> 'ok';

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO schema_migrations (version) VALUES ('0001_init')
ON CONFLICT (version) DO NOTHING;
