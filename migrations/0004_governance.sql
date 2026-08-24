-- Prism 0004 - per-tenant budgets and cost attribution.
-- Idempotent: safe to re-run.

CREATE TABLE IF NOT EXISTS tenant_budgets (
    tenant_id       UUID PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Two caps, because they mean different things. Crossing the soft cap warns
    -- and keeps serving; crossing the hard cap refuses. A single cap forces a
    -- choice between surprising people and cutting them off without warning.
    soft_cap_usd    NUMERIC(14, 6),
    hard_cap_usd    NUMERIC(14, 6),

    -- day | month. The window resets on its own boundary rather than on a
    -- rolling basis, so a tenant can reason about when their budget returns.
    period          TEXT NOT NULL DEFAULT 'month' CHECK (period IN ('day', 'month')),

    -- Denormalized running total for the current period. The traces table is
    -- the source of truth; this exists so the hot path does not aggregate it on
    -- every request, and reconcile_budgets() rebuilds it from traces.
    period_start    DATE NOT NULL DEFAULT date_trunc('month', now())::date,
    spent_usd       NUMERIC(14, 8) NOT NULL DEFAULT 0,

    -- What happens at the hard cap: reject | degrade. Degrading to the cheapest
    -- tier keeps a tenant working at a fraction of the cost, which is usually
    -- better for both sides than a 402.
    hard_cap_action TEXT NOT NULL DEFAULT 'reject'
        CHECK (hard_cap_action IN ('reject', 'degrade')),

    notify_at       NUMERIC(4, 3) NOT NULL DEFAULT 0.8,
    notified_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS tenant_budgets_period_idx ON tenant_budgets (period_start);

-- Append-only ledger of budget events, so "why was I cut off?" has an answer
-- that does not require reconstructing state from traces.
CREATE TABLE IF NOT EXISTS budget_events (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    tenant_id    UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    event        TEXT NOT NULL CHECK (event IN ('soft_cap', 'hard_cap', 'reset', 'degraded')),
    spent_usd    NUMERIC(14, 8) NOT NULL,
    cap_usd      NUMERIC(14, 6),
    detail       JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS budget_events_tenant_idx ON budget_events (tenant_id, created_at DESC);

-- Cost attribution. A view rather than a table: traces already hold every token
-- class, and a second copy would be one more thing to keep in step.
CREATE OR REPLACE VIEW cost_attribution AS
SELECT
    tenant_id,
    date_trunc('day', created_at)::date       AS day,
    model_resolved,
    prompt_version,
    endpoint,
    count(*)                                  AS requests,
    count(*) FILTER (WHERE status = 'ok')     AS successful,
    sum(input_tokens)                         AS uncached_input_tokens,
    sum(cache_read_input_tokens)              AS cached_input_tokens,
    sum(cache_creation_input_tokens)          AS cache_write_tokens,
    sum(output_tokens)                        AS output_tokens,
    sum(thinking_tokens)                      AS thinking_tokens,
    sum(cost_usd)                             AS cost_usd,
    -- Cost per SUCCESSFUL task, not per request. A cheap tier that fails 40% of
    -- the time and retries on an expensive one is not cheap, and only this
    -- denominator shows it.
    CASE WHEN count(*) FILTER (WHERE status = 'ok') > 0
         THEN sum(cost_usd) / count(*) FILTER (WHERE status = 'ok')
    END                                       AS cost_per_successful_task
FROM traces
GROUP BY tenant_id, day, model_resolved, prompt_version, endpoint;

INSERT INTO schema_migrations (version) VALUES ('0004_governance')
ON CONFLICT (version) DO NOTHING;
