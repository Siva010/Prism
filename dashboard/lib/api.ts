/**
 * Read-only client for the gateway's admin API.
 *
 * The dashboard holds no database credentials. It calls the same admin API any
 * other client would, with a tenant-scoped key, so tenant isolation is enforced
 * once — server-side, in the gateway — rather than twice with a chance of the
 * two disagreeing.
 *
 * Every call is allowed to fail. A gateway that is down should render an
 * explanation, not a stack trace, so `fetchAdmin` returns a discriminated result
 * and each page decides what an absent section looks like.
 */

const BASE = process.env.PRISM_URL ?? "http://localhost:8000";
const KEY = process.env.PRISM_API_KEY ?? "";

export type Ok<T> = { ok: true; data: T };
export type Err = { ok: false; error: string; status?: number };
export type Result<T> = Ok<T> | Err;

export async function fetchAdmin<T>(path: string): Promise<Result<T>> {
  try {
    const response = await fetch(`${BASE}${path}`, {
      headers: KEY ? { authorization: `Bearer ${KEY}` } : {},
      // Always live. A cached dashboard showing a healthy circuit breaker that
      // opened two minutes ago is worse than no dashboard.
      cache: "no-store",
    });
    if (!response.ok) {
      const body = await response.text();
      return {
        ok: false,
        status: response.status,
        error: body.slice(0, 300) || response.statusText,
      };
    }
    return { ok: true, data: (await response.json()) as T };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

// --- shapes the admin API returns ----------------------------------------

export type TokenClasses = {
  uncached_input: number;
  cached_input: number;
  cache_write: number;
  output: number;
  thinking: number;
};

export type TraceSummary = {
  id: string;
  created_at: string;
  endpoint: string;
  model_requested: string;
  model_resolved: string;
  stream: boolean;
  status: string;
  http_status: number | null;
  error_type: string | null;
  stop_reason: string | null;
  finish_reason: string | null;
  latency_ms: number | null;
  ttft_ms: number | null;
  gateway_overhead_ms: number | null;
  tokens: TokenClasses;
  cost_usd: string;
  prompt_version: string | null;
};

export type TraceDetail = TraceSummary & {
  cost_breakdown: Record<string, string>;
  request_body: unknown;
  upstream_request: unknown;
  upstream_response: unknown;
  response_body: unknown;
  extra: Record<string, unknown>;
};

export type CacheRow = {
  model: string;
  prompt_version: string | null;
  requests: number;
  uncached_input_tokens: number;
  cached_input_tokens: number;
  cache_write_tokens: number;
  output_tokens: number;
  prefix_cache_hit_rate: number;
  write_read_ratio: number | null;
  cost_usd: string;
};

export type SemanticStats = {
  lookups: number;
  hits: number;
  hit_rate: number;
  avg_hit_similarity: number | null;
  avg_miss_similarity: number | null;
  cost_avoided_usd: string;
};

export type CostRow = {
  day: string;
  model: string;
  prompt_version: string | null;
  requests: number;
  successful: number;
  tokens: TokenClasses;
  cost_usd: string;
  cost_per_successful_task: string | null;
};

export type Interval = {
  point: number;
  low: number;
  high: number;
  confidence: number;
};

export type EvalRun = {
  run_id: string;
  created_at: string;
  dataset: string;
  dataset_size: number;
  candidate: string;
  baseline: string | null;
  metrics: Record<string, Interval>;
  deltas: Record<string, Interval>;
  judge: Record<string, number> | null;
  calibration: { cohens_kappa: number; interpretation: string; trustworthy: boolean } | null;
  cost_usd: string;
  cost_per_success: string | null;
  failures: number;
  regressed: boolean;
  tolerance: number;
  git_sha: string | null;
};

export type BreakerHealth = {
  provider: string;
  state: string;
  consecutive_failures: number;
  failure_rate: number;
  retry_after_s: number;
  opens: number;
  ignored_failures: number;
};

export type RateLimitHealth = {
  enabled: boolean;
  limits?: Record<string, number>;
  reconciliation?: {
    samples: number;
    accuracy: number;
    under_reservation_rate: number;
    worst_under_reservation: number;
  };
  effective_input_capacity?: Record<string, number>;
};

export type RouterInfo = {
  enabled: boolean;
  trained?: boolean;
  mode?: string;
  ladder?: string[];
  break_even: {
    cheap: string;
    expensive: string;
    cost_ratio: number;
    threshold: number;
  }[];
  coefficients?: Record<string, number>;
};

export type PromptGroup = {
  name: string;
  active: string | null;
  versions: {
    version: string;
    content_hash: string;
    chars: number;
    active: boolean;
  }[];
};

export type BudgetStatus = {
  status: string;
  spent_usd: string;
  soft_cap_usd: string | null;
  hard_cap_usd: string | null;
  utilisation: number | null;
  events: { created_at: string; event: string; spent_usd: string }[];
};

export type ListOf<T> = { object: "list"; data: T[] };

// --- formatting ----------------------------------------------------------

export const usd = (value: string | number, digits = 4) =>
  `$${Number(value).toFixed(digits)}`;

export const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;

export const num = (value: number | null | undefined) =>
  value === null || value === undefined ? "—" : value.toLocaleString("en-US");

export const ms = (value: number | null | undefined) =>
  value === null || value === undefined ? "—" : `${value.toLocaleString("en-US")} ms`;

/** An interval, never a bare point — the house style throughout this project. */
export const interval = (i: Interval | undefined, digits = 4) =>
  i ? `${i.point.toFixed(digits)} [${i.low.toFixed(digits)}, ${i.high.toFixed(digits)}]` : "—";
