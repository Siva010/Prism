import { Card, Empty, Tag, Unavailable } from "@/components/ui";
import {
  fetchAdmin,
  num,
  pct,
  usd,
  type CacheRow,
  type ListOf,
  type SemanticStats,
} from "@/lib/api";

export const dynamic = "force-dynamic";

type Calibration = {
  id: string;
  created_at: string;
  embedder: string;
  n_pairs: number;
  auc: number | null;
  max_false_hit_rate: number;
  chosen_threshold: number | null;
  chosen: { false_hit_upper_bound?: number; recall?: number; hit_rate?: number } | null;
  curve: {
    threshold: number;
    hit_rate: number;
    precision: number;
    recall: number;
    false_hit_rate: number;
    false_hit_upper_bound: number;
  }[];
};

export default async function Caching() {
  const [prefix, semantic, calibrations] = await Promise.all([
    fetchAdmin<ListOf<CacheRow>>("/admin/stats/cache"),
    fetchAdmin<SemanticStats>("/admin/stats/semantic-cache"),
    fetchAdmin<ListOf<Calibration>>("/admin/cache/calibrations?limit=1"),
  ]);

  const latest = calibrations.ok ? calibrations.data.data[0] : undefined;

  return (
    <>
      <h1>Caching</h1>
      <p className="lede">
        Two layers with different failure modes. The prefix cache is exact-match
        and cannot return a wrong answer — only cost money when placed badly. The
        semantic cache is fuzzy and <em>can</em>, so it has to justify itself on
        the residual the first layer leaves behind.
      </p>

      {!prefix.ok ? <Unavailable what="Prefix cache statistics" error={prefix} /> : null}

      <h2>Layer 1 — provider prefix cache</h2>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Prompt version</th>
              <th className="num">Requests</th>
              <th className="num">Hit rate</th>
              <th className="num">Write / read</th>
              <th className="num">Cached in</th>
              <th className="num">Written</th>
              <th className="num">Cost</th>
            </tr>
          </thead>
          <tbody>
            {prefix.ok && prefix.data.data.length ? (
              prefix.data.data.map((r, i) => {
                const badPlacement = r.write_read_ratio !== null && r.write_read_ratio > 1;
                return (
                  <tr key={`${r.model}-${r.prompt_version}-${i}`}>
                    <td className="mono">{r.model}</td>
                    <td className="mono">{r.prompt_version ?? "—"}</td>
                    <td className="num">{num(r.requests)}</td>
                    <td className="num">{pct(r.prefix_cache_hit_rate)}</td>
                    <td className="num">
                      {r.write_read_ratio === null ? (
                        "—"
                      ) : (
                        <Tag kind={badPlacement ? "bad" : "ok"}>
                          {r.write_read_ratio.toFixed(2)}
                        </Tag>
                      )}
                    </td>
                    <td className="num">{num(r.cached_input_tokens)}</td>
                    <td className="num">{num(r.cache_write_tokens)}</td>
                    <td className="num">{usd(r.cost_usd)}</td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={8}>
                  <Empty>
                    {prefix.ok
                      ? "No cache statistics yet."
                      : "Not loaded — see above. This is not an empty result."}
                  </Empty>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="note">
        The write/read ratio is the number that catches a misplaced breakpoint.
        Persistently above 1 means the breakpoints sit on volatile content: every
        request pays the ~25% write premium and nothing is ever read back, which
        is strictly <em>worse</em> than placing no breakpoint at all.
      </p>

      <h2>Layer 2 — semantic cache</h2>
      {semantic.ok ? (
        <>
          <div className="cards">
            <Card
              label="Hit rate"
              value={pct(semantic.data.hit_rate)}
              sub={`${semantic.data.hits} of ${semantic.data.lookups} lookups`}
            />
            <Card
              label="Cost avoided"
              value={usd(semantic.data.cost_avoided_usd)}
              sub="on hits, at the prices the misses would have paid"
            />
            <Card
              label="Mean hit similarity"
              value={
                semantic.data.avg_hit_similarity === null
                  ? "—"
                  : semantic.data.avg_hit_similarity.toFixed(4)
              }
            />
            <Card
              label="Mean near-miss"
              value={
                semantic.data.avg_miss_similarity === null
                  ? "—"
                  : semantic.data.avg_miss_similarity.toFixed(4)
              }
              sub="what a threshold re-tune runs on"
            />
          </div>
          <p className="note">
            A hit rate on its own says nothing about whether the threshold is in
            the right place, which is why the near-miss distribution sits beside
            it. The headline number for this layer is not its hit rate but its{" "}
            <em>incremental</em> saving over prefix caching alone — see{" "}
            <code>scripts/replay.py</code>, which runs the corpus through all
            three configurations.
          </p>
        </>
      ) : (
        <Unavailable what="Semantic cache statistics" error={semantic} />
      )}

      <h2>Threshold calibration</h2>
      {latest ? (
        <>
          <div className="cards">
            <Card label="Labelled pairs" value={num(latest.n_pairs)} sub={latest.embedder} />
            <Card
              label="AUC"
              value={latest.auc === null ? "—" : latest.auc.toFixed(4)}
              sub="threshold-independent separability"
            />
            <Card
              label="Chosen threshold"
              value={
                latest.chosen_threshold === null
                  ? "none viable"
                  : latest.chosen_threshold.toFixed(3)
              }
              sub={`ceiling ${pct(latest.max_false_hit_rate)} false hits`}
            />
            <Card
              label="False-hit upper bound"
              value={pct(latest.chosen?.false_hit_upper_bound)}
              sub="95% CI, not the point estimate"
            />
          </div>

          <div className="scroll" style={{ marginTop: 14 }}>
            <table>
              <thead>
                <tr>
                  <th className="num">Threshold</th>
                  <th className="num">Hit rate</th>
                  <th className="num">Precision</th>
                  <th className="num">Recall</th>
                  <th className="num">False-hit rate</th>
                  <th className="num">Upper bound</th>
                </tr>
              </thead>
              <tbody>
                {latest.curve.map((p) => (
                  <tr
                    key={p.threshold}
                    style={
                      p.threshold === latest.chosen_threshold
                        ? { background: "color-mix(in srgb, #4ec9a5 12%, transparent)" }
                        : undefined
                    }
                  >
                    <td className="num mono">{p.threshold.toFixed(3)}</td>
                    <td className="num">{pct(p.hit_rate)}</td>
                    <td className="num">{pct(p.precision)}</td>
                    <td className="num">{pct(p.recall)}</td>
                    <td className="num">{pct(p.false_hit_rate)}</td>
                    <td className="num">{pct(p.false_hit_upper_bound)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            The operating point is chosen on the <em>upper bound</em> of the
            false-hit interval, not the point estimate. On a small labelled set
            the estimate is often exactly 0.0, and treating that as proof of
            safety is how a cache ships with an unmeasured error rate: 0 false
            hits in 20 served means the true rate is under roughly 16%, not zero.
          </p>
        </>
      ) : (
        <p className="note">
          {calibrations.ok
            ? "No calibration runs stored. Until one exists the semantic cache stays off — a threshold nobody measured is not a threshold. Run "
            : "Calibration history could not be loaded, so nothing is shown here. Run "}
          <code>scripts/calibrate_cache.py</code>.
        </p>
      )}
    </>
  );
}
