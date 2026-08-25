import type { Err, Result } from "@/lib/api";

/**
 * A section that failed to load says why, and says what is *not* being shown.
 *
 * The alternative — rendering zeros — is worse than useless here: a dashboard
 * that shows a $0.00 spend and a healthy circuit breaker because the gateway is
 * unreachable is actively misleading, and this project's whole argument is that
 * an unmeasured number should never look like a measured one.
 */
export function Unavailable({ what, error }: { what: string; error: Err }) {
  return (
    <div className="banner">
      <strong>{what} unavailable.</strong>{" "}
      {error.status ? `HTTP ${error.status}. ` : ""}
      {error.error}
      <div className="note">
        Nothing is being shown for this section — not zero, not stale. Check that
        the gateway is running at <code>PRISM_URL</code> and that{" "}
        <code>PRISM_API_KEY</code> is a valid tenant key.
      </div>
    </div>
  );
}

export function Card({
  label,
  value,
  sub,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
}) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  );
}

export function Tag({
  kind = "",
  children,
}: {
  kind?: "ok" | "warn" | "bad" | "";
  children: React.ReactNode;
}) {
  return <span className={`tag ${kind}`}>{children}</span>;
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="empty">{children}</div>;
}

/** Narrow a Result at a type level, so pages can early-return on failure. */
export function failed<T>(r: Result<T>): r is Err {
  return !r.ok;
}

const TOKEN_COLOURS: Record<string, string> = {
  uncached_input: "#6ea8fe",
  cached_input: "#4ec9a5",
  cache_write: "#e3b341",
  output: "#b78cf0",
  thinking: "#8b97a6",
};

/**
 * The five token classes as one bar.
 *
 * Shown as classes rather than a total on purpose: they price differently, and
 * `total_tokens x unit_price` is wrong by a large margin once caching is on.
 */
export function TokenBar({ tokens }: { tokens: Record<string, number> }) {
  const entries = Object.entries(tokens).filter(([, v]) => v > 0);
  const total = entries.reduce((sum, [, v]) => sum + v, 0);
  if (!total) return <span className="sub">no tokens</span>;

  return (
    <div>
      <div className="bar">
        {entries.map(([name, value]) => (
          <span
            key={name}
            style={{
              width: `${(value / total) * 100}%`,
              background: TOKEN_COLOURS[name] ?? "#555",
            }}
            title={`${name}: ${value.toLocaleString("en-US")}`}
          />
        ))}
      </div>
      <div className="legend">
        {entries.map(([name, value]) => (
          <span key={name}>
            <i style={{ background: TOKEN_COLOURS[name] ?? "#555" }} />
            {name.replace(/_/g, " ")} {value.toLocaleString("en-US")}
          </span>
        ))}
      </div>
    </div>
  );
}
