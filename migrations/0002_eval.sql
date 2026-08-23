-- Prism 0002 — evaluation runs and per-example results.
-- Idempotent: safe to re-run.

CREATE TABLE IF NOT EXISTS eval_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              TEXT NOT NULL UNIQUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,

    dataset             TEXT NOT NULL,
    dataset_size        INTEGER NOT NULL DEFAULT 0,

    candidate_name      TEXT NOT NULL,
    candidate_config    JSONB NOT NULL DEFAULT '{}'::jsonb,
    baseline_name       TEXT,
    baseline_config     JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Bootstrap intervals, per metric: {"exact_match": {"point":..,"low":..,"high":..}}
    -- Point estimates are never stored without their interval.
    metrics             JSONB NOT NULL DEFAULT '{}'::jsonb,
    deltas              JSONB NOT NULL DEFAULT '{}'::jsonb,

    judge_model         TEXT,
    judge_summary       JSONB NOT NULL DEFAULT '{}'::jsonb,
    judge_delta         JSONB,
    -- Cohen's kappa against human labels, carried on every run that used a
    -- judge. A win rate without it is not a measurement.
    calibration         JSONB,

    cost_usd            NUMERIC(14, 8) NOT NULL DEFAULT 0,
    cost_per_success    NUMERIC(14, 8),
    failures            INTEGER NOT NULL DEFAULT 0,

    -- The gate decision, frozen at run time. Recomputing it later against a
    -- different tolerance would rewrite history.
    regressed           BOOLEAN NOT NULL DEFAULT FALSE,
    tolerance           DOUBLE PRECISION NOT NULL DEFAULT 0,

    git_sha             TEXT,
    ci_run_url          TEXT,
    extra               JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS eval_runs_created_idx ON eval_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS eval_runs_dataset_idx ON eval_runs (dataset, created_at DESC);

CREATE TABLE IF NOT EXISTS eval_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    example_id          TEXT NOT NULL,
    arm                 TEXT NOT NULL,            -- candidate | baseline

    response_text       TEXT,
    model               TEXT,
    cost_usd            NUMERIC(14, 8) NOT NULL DEFAULT 0,
    latency_ms          INTEGER,
    -- Joins an eval result back to the gateway trace that produced it, so a
    -- surprising score can be opened in the trace explorer.
    trace_id            TEXT,
    error               TEXT,

    scores              JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Pairwise judge output, when this example was judged.
    judge_verdict       TEXT,
    judge_reason        TEXT,
    judge_raw_forward   TEXT,
    judge_raw_reverse   TEXT,

    UNIQUE (run_id, example_id, arm)
);

CREATE INDEX IF NOT EXISTS eval_results_run_idx ON eval_results (run_id);
CREATE INDEX IF NOT EXISTS eval_results_example_idx ON eval_results (example_id);

-- Human labels for judge calibration. Kept in the database rather than only in
-- a file so kappa can be recomputed whenever the judge model or prompt changes,
-- without re-labelling.
CREATE TABLE IF NOT EXISTS human_labels (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    example_id          TEXT NOT NULL,
    label               TEXT NOT NULL CHECK (label IN ('A', 'B', 'tie')),
    annotator           TEXT NOT NULL DEFAULT 'human',
    notes               TEXT NOT NULL DEFAULT '',
    -- Which comparison was labelled: labels are only meaningful against the
    -- exact pair of responses the annotator saw.
    source_run_id       TEXT,
    UNIQUE (example_id, annotator, source_run_id)
);

CREATE INDEX IF NOT EXISTS human_labels_example_idx ON human_labels (example_id);

INSERT INTO schema_migrations (version) VALUES ('0002_eval')
ON CONFLICT (version) DO NOTHING;
