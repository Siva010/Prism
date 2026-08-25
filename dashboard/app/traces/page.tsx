import Link from "next/link";

import { Empty, Tag, Unavailable } from "@/components/ui";
import { fetchAdmin, ms, usd, type ListOf, type TraceSummary } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Traces() {
  const traces = await fetchAdmin<ListOf<TraceSummary>>("/admin/traces?limit=100");

  return (
    <>
      <h1>Traces</h1>
      <p className="lede">
        Every request, with its five token classes and what the gateway itself
        cost on top of the upstream call.
      </p>

      {!traces.ok ? <Unavailable what="Traces" error={traces} /> : null}

      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Requested</th>
              <th>Served</th>
              <th>Status</th>
              <th className="num">Latency</th>
              <th className="num">TTFT</th>
              <th className="num">Overhead</th>
              <th className="num">Cached in</th>
              <th className="num">Cost</th>
            </tr>
          </thead>
          <tbody>
            {traces.ok && traces.data.data.length ? (
              traces.data.data.map((t) => (
                <tr key={t.id}>
                  <td>
                    <Link href={`/traces/${t.id}`}>
                      {new Date(t.created_at).toLocaleTimeString()}
                    </Link>
                  </td>
                  <td className="mono">{t.model_requested}</td>
                  <td className="mono">{t.model_resolved}</td>
                  <td>
                    <Tag kind={t.status === "ok" ? "ok" : t.status === "client_disconnect" ? "warn" : "bad"}>
                      {t.status === "ok" ? (t.stream ? "stream" : "ok") : t.error_type ?? t.status}
                    </Tag>
                  </td>
                  <td className="num">{ms(t.latency_ms)}</td>
                  <td className="num">{ms(t.ttft_ms)}</td>
                  <td className="num">{ms(t.gateway_overhead_ms)}</td>
                  <td className="num">{t.tokens.cached_input.toLocaleString("en-US")}</td>
                  <td className="num">{usd(t.cost_usd, 6)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={9}>
                  <Empty>
                    {traces.ok
                      ? "No traces yet. Send a request through /v1/chat/completions."
                      : "Could not reach the gateway."}
                  </Empty>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="note">
        Overhead is total latency minus the upstream call. It is not clamped at
        zero: a negative value means the two clocks disagree, and hiding that
        behind a 0 would make a broken measurement look like a fast one. It is
        blank for streams, where there is no bracketed upstream call to subtract.
      </p>
    </>
  );
}
