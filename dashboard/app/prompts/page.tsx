import { Tag, Unavailable } from "@/components/ui";
import { fetchAdmin, type ListOf, type PromptGroup, type RouterInfo } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Prompts() {
  const [prompts, router] = await Promise.all([
    fetchAdmin<ListOf<PromptGroup>>("/admin/prompts"),
    fetchAdmin<RouterInfo>("/admin/router"),
  ]);

  return (
    <>
      <h1>Prompts &amp; routing</h1>
      <p className="lede">
        A prompt version is a cache-key input and a prefix-cache breakpoint
        boundary, so promoting one has a predictable, measurable cache cost rather
        than a mysterious hit-rate cliff.
      </p>

      {!prompts.ok ? <Unavailable what="Prompt registry" error={prompts} /> : null}

      {prompts.ok
        ? prompts.data.data.map((group) => {
            const newest = group.versions[group.versions.length - 1];
            const pending = newest && !newest.active;
            const activeHash = group.versions.find((v) => v.active)?.content_hash ?? "—";
            return (
              <div key={group.name} style={{ marginBottom: 22 }}>
                <h2 style={{ marginBottom: 8 }}>
                  <span className="mono">{group.name}</span>{" "}
                  {pending ? (
                    <Tag kind="warn">pending the gate</Tag>
                  ) : (
                    <Tag kind="ok">current</Tag>
                  )}
                </h2>
                <div className="scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Version</th>
                        <th>Content hash</th>
                        <th className="num">Chars</th>
                        <th>State</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.versions.map((v) => (
                        <tr key={v.version}>
                          <td className="mono">{v.version}</td>
                          <td className="mono">{v.content_hash}</td>
                          <td className="num">{v.chars.toLocaleString("en-US")}</td>
                          <td>
                            {v.active ? (
                              <Tag kind="ok">serving traffic</Tag>
                            ) : (
                              <Tag>inactive</Tag>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {pending ? (
                  <p className="note">
                    Promoting <span className="mono">{newest.version}</span> changes the
                    content hash from <span className="mono">{activeHash}</span> to{" "}
                    <span className="mono">{newest.content_hash}</span>. Both are
                    cache-key inputs, so the first request against every affected prefix
                    becomes a cache write rather than a read.
                  </p>
                ) : null}
              </div>
            );
          })
        : null}

      <h2>Router break-even thresholds</h2>
      {router.ok ? (
        <>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Downgrade</th>
                  <th className="num">Cost ratio</th>
                  <th className="num">Route down when P(success) &gt;</th>
                </tr>
              </thead>
              <tbody>
                {router.data.break_even.map((r) => (
                  <tr key={`${r.cheap}-${r.expensive}`}>
                    <td className="mono">
                      {r.cheap} → {r.expensive}
                    </td>
                    <td className="num">{r.cost_ratio.toFixed(3)}</td>
                    <td className="num mono">{r.threshold.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            Not tuned. With <code>p</code> the probability the cheap tier copes and a
            failure retried upstream, routing down wins exactly when{" "}
            <code>c + (1−p)e &lt; e</code>, i.e. <code>p &gt; c/e</code>. The threshold{" "}
            <em>is</em> the cost ratio, and it moves with the price list rather than with
            anyone&apos;s judgement.
          </p>

          {router.data.coefficients ? (
            <>
              <h2>Router coefficients</h2>
              <div className="scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th className="num">Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(router.data.coefficients).map(([name, weight]) => (
                      <tr key={name}>
                        <td className="mono">{name}</td>
                        <td className="num mono">
                          {weight >= 0 ? "+" : ""}
                          {weight.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="note">
                Positive means the cheap tier is likely to cope. These are readable
                because the router is a logistic regression — &ldquo;why was this routed
                up?&rdquo; has an answer that fits in a sentence, which is most of the
                argument for not fine-tuning something here.
              </p>
            </>
          ) : (
            <p className="note">
              No trained router. Requests naming <code>prism-auto</code> go to the top of
              the ladder — falling back to the cheap tier would silently degrade quality
              the moment a model file went missing.
            </p>
          )}
        </>
      ) : (
        <Unavailable what="Router" error={router} />
      )}
    </>
  );
}
