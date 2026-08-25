import { Empty, Tag, Unavailable } from "@/components/ui";
import { fetchAdmin, interval, pct, usd, type EvalRun, type ListOf } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Evals() {
  const runs = await fetchAdmin<ListOf<EvalRun>>("/admin/eval/runs?limit=25");

  return (
    <>
      <h1>Evaluation</h1>
      <p className="lede">
        Every metric carries its bootstrap confidence interval. On a 200-example
        set a 3% difference is usually noise, so the interval is what decides
        whether a change shipped or regressed — never the point estimate.
      </p>

      {!runs.ok ? <Unavailable what="Eval runs" error={runs} /> : null}

      {runs.ok && runs.data.data.length ? (
        runs.data.data.map((run) => (
          <div key={run.run_id} style={{ marginBottom: 26 }}>
            <h2 style={{ marginBottom: 6 }}>
              <span className="mono">{run.candidate}</span>
              {run.baseline ? (
                <>
                  {" vs "}
                  <span className="mono">{run.baseline}</span>
                </>
              ) : null}{" "}
              <Tag kind={run.regressed ? "bad" : "ok"}>
                {run.regressed ? "regression" : "passed"}
              </Tag>
            </h2>
            <p className="note" style={{ marginTop: 0 }}>
              {new Date(run.created_at).toLocaleString()} · {run.dataset_size} examples ·{" "}
              {usd(run.cost_usd)}
              {run.cost_per_success
                ? ` · ${usd(run.cost_per_success, 6)} per successful task`
                : ""}
              {run.failures ? ` · ${run.failures} failed calls` : ""}
              {run.git_sha ? ` · ${run.git_sha.slice(0, 8)}` : ""}
            </p>

            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th className="num">Candidate (95% CI)</th>
                    <th className="num">Delta (95% CI)</th>
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(run.metrics).map(([metric, value]) => {
                    const delta = run.deltas?.[metric];
                    const regressed = delta ? delta.high < -Math.abs(run.tolerance) : false;
                    const improved = delta ? delta.low > 0 : false;
                    return (
                      <tr key={metric}>
                        <td>{metric.replace(/_/g, " ")}</td>
                        <td className="num mono">{interval(value)}</td>
                        <td className="num mono">{delta ? interval(delta) : "—"}</td>
                        <td>
                          {!delta ? (
                            "—"
                          ) : regressed ? (
                            <Tag kind="bad">regression</Tag>
                          ) : improved ? (
                            <Tag kind="ok">improved</Tag>
                          ) : (
                            <Tag>no change</Tag>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {run.judge ? (
              <p className="note">
                Judge: {run.judge.candidate_wins}W / {run.judge.baseline_wins}L /{" "}
                {run.judge.ties}T, {run.judge.inconsistent} inconsistent · position-bias
                rate {pct(run.judge.position_bias_rate)}
                {run.calibration ? (
                  <>
                    {" · kappa "}
                    {run.calibration.cohens_kappa.toFixed(3)} (
                    {run.calibration.interpretation})
                    {!run.calibration.trustworthy ? (
                      <>
                        {" — "}
                        <strong>below substantial agreement, treat as directional</strong>
                      </>
                    ) : null}
                  </>
                ) : (
                  <>
                    {" — "}
                    <strong>
                      uncalibrated: no human labels, so these verdicts carry no known
                      agreement rate
                    </strong>
                  </>
                )}
              </p>
            ) : null}
          </div>
        ))
      ) : (
        <div className="scroll">
          <Empty>
            {runs.ok
              ? "No eval runs stored yet. Run scripts/eval.py."
              : "Could not reach the gateway."}
          </Empty>
        </div>
      )}

      <p className="note">
        The gate fails only when a metric&apos;s <em>entire</em> interval sits below
        tolerance. One that fired on the point estimate would fail roughly half of all
        no-op changes and be switched off within a week.
      </p>
    </>
  );
}
