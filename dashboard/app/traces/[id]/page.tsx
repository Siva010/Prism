import Link from "next/link";

import { TokenBar, Unavailable } from "@/components/ui";
import { fetchAdmin, ms, usd, type TraceDetail } from "@/lib/api";

export const dynamic = "force-dynamic";

function Json({ title, value }: { title: string; value: unknown }) {
  if (value === null || value === undefined) return null;
  return (
    <>
      <h2>{title}</h2>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </>
  );
}

export default async function TraceDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const trace = await fetchAdmin<TraceDetail>(`/admin/traces/${id}`);

  if (!trace.ok) {
    return (
      <>
        <h1>Trace</h1>
        <Unavailable what="Trace" error={trace} />
        <Link href="/traces">Back to traces</Link>
      </>
    );
  }

  const t = trace.data;
  return (
    <>
      <h1>Trace</h1>
      <p className="lede mono">{t.id}</p>

      <div className="cards">
        <div className="card">
          <div className="label">Model</div>
          <div className="value" style={{ fontSize: 15 }}>{t.model_resolved}</div>
          <div className="sub">requested as {t.model_requested}</div>
        </div>
        <div className="card">
          <div className="label">Cost</div>
          <div className="value">{usd(t.cost_usd, 6)}</div>
          <div className="sub">{t.prompt_version ?? "no prompt version"}</div>
        </div>
        <div className="card">
          <div className="label">Latency</div>
          <div className="value">{ms(t.latency_ms)}</div>
          <div className="sub">
            {t.ttft_ms !== null ? `${t.ttft_ms} ms to first token` : "non-streaming"}
          </div>
        </div>
        <div className="card">
          <div className="label">Stop reason</div>
          <div className="value" style={{ fontSize: 15 }}>{t.stop_reason ?? "—"}</div>
          <div className="sub">returned as {t.finish_reason ?? "—"}</div>
        </div>
      </div>

      <h2>Token classes</h2>
      <TokenBar tokens={t.tokens as unknown as Record<string, number>} />
      <p className="note">
        Priced separately: cache reads are ~10% of the input rate and cache
        writes ~125% of it. A single total would be wrong in both directions.
      </p>

      {Object.keys(t.cost_breakdown ?? {}).length ? (
        <>
          <h2>Cost breakdown</h2>
          <div className="scroll">
            <table>
              <tbody>
                {Object.entries(t.cost_breakdown).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k.replace(/_/g, " ")}</td>
                    <td className="num mono">{usd(v, 8)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}

      <Json title="Gateway decisions" value={t.extra} />
      <Json title="Client request" value={t.request_body} />
      <Json title="Upstream request (translated)" value={t.upstream_request} />
      <Json title="Upstream response" value={t.upstream_response} />
      <Json title="Client response (translated back)" value={t.response_body} />

      <p className="note">
        <Link href="/traces">Back to traces</Link>
      </p>
    </>
  );
}
