import { Card, Tag, Unavailable } from "@/components/ui";
import {
  fetchAdmin,
  pct,
  usd,
  type BreakerHealth,
  type BudgetStatus,
  type CacheRow,
  type ListOf,
  type RateLimitHealth,
  type SemanticStats,
} from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Overview() {
  const [cache, semantic, health, limits, budget] = await Promise.all([
    fetchAdmin<ListOf<CacheRow>>("/admin/stats/cache"),
    fetchAdmin<SemanticStats>("/admin/stats/semantic-cache"),
    fetchAdmin<ListOf<BreakerHealth>>("/admin/health/providers"),
    fetchAdmin<RateLimitHealth>("/admin/health/rate-limits"),
    fetchAdmin<BudgetStatus>("/admin/budget"),
  ]);

  const rows = cache.ok ? cache.data.data : [];
  const totals = rows.reduce(
    (acc, r) => ({
      requests: acc.requests + r.requests,
      cost: acc.cost + Number(r.cost_usd),
      cached: acc.cached + r.cached_input_tokens,
      written: acc.written + r.cache_write_tokens,
      uncached: acc.uncached + r.uncached_input_tokens,
    }),
    { requests: 0, cost: 0, cached: 0, written: 0, uncached: 0 },
  );
  const totalInput = totals.cached + totals.written + totals.uncached;
  const hitRate = totalInput ? totals.cached / totalInput : 0;
  const writeRead = totals.cached ? totals.written / totals.cached : null;

  return (
    <>
      <h1>Overview</h1>
      <p className="lede">
        Every number here is derived from the trace table, which stores the five
        token classes separately and never a total.
      </p>

      {!cache.ok ? <Unavailable what="Cache statistics" error={cache} /> : null}

      <div className="cards">
        <Card label="Requests" value={totals.requests.toLocaleString("en-US")} />
        <Card label="Spend" value={usd(totals.cost)} />
        <Card
          label="Prefix cache hit rate"
          value={pct(hitRate)}
          sub={`${totals.cached.toLocaleString("en-US")} tokens read from cache`}
        />
        <Card
          label="Write / read ratio"
          value={writeRead === null ? "—" : writeRead.toFixed(2)}
          sub={
            writeRead !== null && writeRead > 1
              ? "above 1: breakpoints are on volatile content"
              : "below 1: breakpoints are paying back"
          }
        />
        {semantic.ok ? (
          <Card
            label="Semantic cache"
            value={pct(semantic.data.hit_rate)}
            sub={`${semantic.data.hits} of ${semantic.data.lookups} lookups`}
          />
        ) : null}
        {budget.ok && budget.data.hard_cap_usd ? (
          <Card
            label="Budget"
            value={pct(budget.data.utilisation)}
            sub={`${usd(budget.data.spent_usd, 2)} of ${usd(budget.data.hard_cap_usd, 2)}`}
          />
        ) : null}
      </div>

      <h2>Throughput headroom</h2>
      {limits.ok && limits.data.enabled ? (
        <>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Prefix hit rate</th>
                  <th className="num">Effective input tokens / minute</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(limits.data.effective_input_capacity ?? {}).map(
                  ([key, value]) => (
                    <tr key={key}>
                      <td>{key.replace("hit_rate_", "")}%</td>
                      <td className="num">{Math.round(value).toLocaleString("en-US")}</td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
          <p className="note">
            Cache reads do not count toward the input-token limit, so the prefix
            cache is a throughput multiplier and not only a cost saver:{" "}
            <code>effective ITPM = limit / (1 − hit_rate)</code>.
            {limits.data.reconciliation ? (
              <>
                {" "}
                Reservation accuracy{" "}
                {limits.data.reconciliation.accuracy.toFixed(2)} over{" "}
                {limits.data.reconciliation.samples} samples, with{" "}
                {pct(limits.data.reconciliation.under_reservation_rate)}{" "}
                under-reserved — those are the requests that can still trip a 429.
              </>
            ) : null}
          </p>
        </>
      ) : (
        <p className="note">
          Rate limiting is disabled, so the provider&apos;s own 429s are the only
          backstop. Set <code>PRISM_RATE_LIMIT_ENABLED=true</code> (needs Redis).
        </p>
      )}

      <h2>Provider health</h2>
      {health.ok && health.data.data.length ? (
        <>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>State</th>
                  <th className="num">Consecutive failures</th>
                  <th className="num">Opens</th>
                  <th className="num">429s ignored</th>
                </tr>
              </thead>
              <tbody>
                {health.data.data.map((b) => (
                  <tr key={b.provider}>
                    <td>{b.provider}</td>
                    <td>
                      <Tag kind={b.state === "closed" ? "ok" : b.state === "open" ? "bad" : "warn"}>
                        {b.state}
                      </Tag>
                    </td>
                    <td className="num">{b.consecutive_failures}</td>
                    <td className="num">{b.opens}</td>
                    <td className="num">{b.ignored_failures}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            &ldquo;429s ignored&rdquo; counts quota errors that were correctly{" "}
            <em>not</em> treated as provider ill-health. Quota belongs to the
            account, not the route: counting them would open every circuit during
            a quota burst and stop traffic the gateway could have served.
          </p>
        </>
      ) : (
        <p className="note">No provider health available.</p>
      )}
    </>
  );
}
