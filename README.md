# Prism

A Claude-native LLM gateway: OpenAI-compatible ingress, Anthropic Messages API
egress, layered caching, a two-axis adaptive router, and a CI-gated evaluation
harness.

**Status: complete, 12 of 12 weeks.** Proxy, protocol translation (streaming
and not), five-class cost model, trace persistence, evaluation harness, CI
quality gate, both cache layers, the two-axis router, resilience and
governance, and the dashboard.

One limit stated up front, because it changes how every number below should be
read: **no request in this repository has ever reached the real Anthropic API.**
See [Verified against what](#verified-against-what).

---

## Quick start

```bash
docker compose up -d
```

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Then mint a tenant and
an API key:

```bash
python scripts/create_tenant.py acme "Acme Corp"
```

Run it:

```bash
uvicorn prism.main:app --reload --app-dir src
```

Point any OpenAI client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="prism_sk_...")
client.chat.completions.create(
    model="gpt-4o",  # mapped onto the Claude ladder
    messages=[{"role": "user", "content": "capital of France?"}],
)
```

Tests, lint, and types:

```bash
pytest && ruff check . && mypy src/prism
```

Most of the suite runs against fakes and needs nothing running. The tests in
`tests/test_integration_db.py` need Postgres and skip cleanly without it — they
cover the things whose failure mode only appears against a real server, which is
a category this project learned about the hard way (see [Verified
against](#verified-against-what)).

Run the dashboard (reads the admin API; needs a tenant key):

```bash
cd dashboard && npm install && npm run dev
```

Run the eval suite against a running gateway:

```bash
python scripts/eval.py --dataset datasets/golden/v1.jsonl --candidate v2 --baseline v1
```

It exits non-zero on a regression, which is the entire interface the CI gate
needs. To gate whatever the registry says is pending:

```bash
python scripts/eval.py --auto --human-labels datasets/golden/human_labels.jsonl
```

Ship a prompt change by adding `prompts/<name>/v<N+1>.md` and opening a PR; the
gate scores it against the active version. Promote (or roll back) with:

```bash
python scripts/promote.py assistant v2
```

---

## What is built

| Area | State |
|---|---|
| OpenAI `/v1/chat/completions` ingress | done |
| OpenAI → Anthropic request translation | done |
| Anthropic → OpenAI response translation | done |
| Five-token-class cost model | done |
| Trace persistence (Postgres) + admin read API | done |
| Hashed per-tenant API keys | done |
| Error taxonomy with the 429/529 split | done |
| SSE streaming: block reassembly, tee-and-buffer, split timeouts | done |
| Golden dataset, programmatic metrics, bootstrap CIs | done |
| Pairwise LLM-as-judge with position swapping + kappa calibration | done |
| Versioned prompt registry with promote/rollback | done |
| CI regression gate (required check) | done |
| Prefix-cache breakpoint policy + placement report | done |
| Local token estimator + calibration harness | done |
| Semantic cache (pgvector/HNSW), scoped and calibrated | done |
| Three-configuration replay + threshold ROC sweep | done |
| Exact-match Redis layer | week 10 (with the rate-limit buckets) |
| Two-axis router (tier + reasoning budget) | done |
| Three-dimensional token-aware rate limiting (Redis/Lua) | done |
| Circuit breakers, failover, hedging | done |
| Per-tenant budgets + cost attribution | done |
| Next.js dashboard: traces, caching, evaluation, prompts, routing | done |
| Dashboard | week 12 |

---

---

---

## Verified against what

Worth being explicit, because "the tests pass" and "this works" are different
claims.

| | state |
|---|---|
| Translation, streaming, cost model, eval harness, cache logic, router | unit-tested against fakes |
| Migrations 0001–0004, pgvector round-trip, HNSW index, scope isolation in SQL, budgets, cost attribution | verified against real Postgres |
| Rate-limit Lua, including atomicity under 20-way concurrency | verified against real Redis |
| Dashboard rendering, every view plus the gateway-unreachable path | verified in a browser against a stub serving the admin API contract |
| Semantic-cache calibration and the three-configuration replay | measured with the real `bge-small-en-v1.5` |
| **Anthropic Messages API** | **never called.** No request in this repo has reached a real provider. |

That last row is the honest limit of everything here. The gateway has been
exercised end-to-end against a scripted upstream, and the translation is written
against the documented wire format, but the first real request is still ahead.

One bug worth recording, because it is the argument for the integration tests:
`scripts/migrate.py` was **broken from week 1 to week 9 and nothing noticed**.
Migration 0001 was applied by the Postgres container's initdb entrypoint, so the
runner itself never ran; migrations 0002 and 0003 would have failed on first use.
asyncpg rejects multi-statement SQL through a prepared statement, and every path
through SQLAlchemy — `text()` and `exec_driver_sql()` alike — hands it one. The
fix is to go to asyncpg's own `execute()`, which uses the simple query protocol.

## The result

The headline number this project exists to produce, measured on 2026-08-24 with
`bge-small-en-v1.5` and a synthetic 100-request corpus:

| configuration | cost | per request | saving |
|---|---|---|---|
| no cache | $2.5000 | $0.025000 | — |
| prefix only | $1.1672 | $0.011672 | **53.3%**, at zero correctness risk |
| prefix + semantic | $0.7992 | $0.007992 | **31.5% incremental**, 68.0% total |

At threshold 0.91 the semantic layer served 32 of 100 requests. Labelling every
one of those hits against the ground-truth pair set by hand:

- 24 were byte-identical queries (trivially correct)
- 8 were labelled-equivalent rephrasings (correct)
- **0 were false hits**, 95% CI **0% – 10.7%**

Reproduce it:

```bash
python scripts/calibrate_cache.py --pairs datasets/cache_pairs.jsonl --max-false-hit-rate 0.40
```
```bash
python scripts/replay.py --from-pairs datasets/cache_pairs.jsonl --threshold 0.91 --audit-out hits.jsonl
```

### What these numbers are not

Four caveats, none of which are optional when quoting the figures above.

1. **The corpus is synthetic.** It is built from the labelled pair file, where
   ~40% of pairs are equivalent by construction. Real traffic repeats itself far
   less, so the 31.5% incremental saving is an upper bound on a favourable
   distribution, not a production measurement.
2. **The prefix cache state is simulated.** Provider cache contents are not
   observable from outside, so replaying live three times would not isolate the
   variable and would cost three times as much. The simulation runs real
   breakpoint placement and the real cost model over a TTL window.
3. **A 0% false-hit rate on 32 served responses is not proof of safety.** The
   Wilson upper bound is 10.7%. That interval is the number to defend a threshold
   with, not the point estimate.
4. **At a 1% false-hit ceiling, no threshold qualifies at all.** With 50 labelled
   pairs the interval is wider than the ceiling, so `calibrate_cache.py` exits 1
   and says the cache should stay off. Getting to a defensible 1% needs ~200
   pairs. That is the honest state of this layer today, and it is why
   `semantic_cache_enabled` defaults to **false**.

The AUC is **0.910**, which is the encouraging part: the classes separate well,
so the limit here is labelled-set size rather than the embedding model.

## Design notes

Decisions here that were not obvious, and the reasoning behind them. This section
is the deliverable, not the code.

### The translation is not a field rename

Anthropic ships an OpenAI-compatibility shim. Routing through it would mean
building nothing, so the translation is written out. Four differences do real
work:

* **System prompts move out of the message array** into a top-level field. OpenAI
  allows a system message anywhere; Anthropic has exactly one slot rendered before
  all messages. Hoisting a mid-conversation system message is correct but lossy —
  the instruction ends up in front of the history it was meant to follow — so it
  is recorded as a translation warning on the trace rather than performed
  silently.
* **Tool calls and results move from message fields into content blocks.**
  OpenAI's `tool_calls` array becomes `tool_use` blocks, and `role: "tool"`
  messages become `tool_result` blocks inside a *user* turn. Tool arguments are a
  JSON string on one side and a JSON object on the other, so this is a parse, not
  a copy — and malformed arguments are rejected here rather than forwarded.
* **Turns must alternate.** Several OpenAI messages collapse into one Anthropic
  turn. Parallel tool results in particular must land in a *single* user message;
  emitting one message each breaks alternation and trains the model out of making
  parallel calls.
* **`max_tokens` is optional upstream of the gateway and required downstream**, so
  it is defaulted and clamped to the resolved model's ceiling.

### The routing ladder does not have a uniform request surface

`temperature` and `top_p` are accepted by Haiku 4.5 and rejected with a 400 by
Opus 5 and Sonnet 5. A gateway that forwards them has a client request that
succeeds or fails depending on which tier the router happened to pick — the
routing experiment would be measuring the gateway's bugs. Prism keeps a
capability table in `registry.py` and drops what the target model cannot accept,
recording the drop on the trace.

### Cost is never a total

Five token classes price differently: uncached input, cached input, cache writes,
output, and thinking (billed inside output). `total_tokens × unit_price` is wrong
in both directions once caching is on — it overcharges cache reads by 10× and
undercharges cache writes by 25%. `cost.py` prices each class and keeps
`naive_cost()` alongside so the dashboard can show the size of the gap.

A worked example from the test suite: 100K uncached + 900K cached input + 1K
output on Opus 5 costs **$0.975**, where the naive model reports **$5.025** — a
5.2× overstatement.

The trace table stores the classes, never a total, and every arithmetic step uses
`Decimal`. Cost breakdowns serialize to JSONB as strings, because a round-trip
through float loses cents at aggregate scale.

### `429` and `529` are different events

A `429` means *this account* is out of quota: back off, respect `retry-after`, and
do **not** fail over — another route carries the same exhausted quota. A `529`
means the *provider* is over capacity: quota is irrelevant, fail over or wait.
They are separate members of `ErrorKind`, only capacity-class errors report
`counts_toward_circuit_breaker`, and both stay distinct all the way into the
trace table. Conflating them is the most common bug in hand-rolled client code.

### The write/read ratio is the cache metric that matters

`GET /admin/stats/cache` groups traces by model and prompt version and reports
cache writes per cache read. Persistently above ~1 means the breakpoints sit on
volatile content: every request pays the write premium and nothing is ever read
back, which is strictly *worse* than placing no breakpoint at all. The endpoint
exists before the caching work does, so week 7 has a measurement to aim at
instead of a hit-rate number to admire.

### The stream buffer is a state machine, not a string

OpenAI's stream is flat: concatenate `delta.content` and you have the answer.
Anthropic's is *block-structured* — `message_start`, then a
`content_block_start` / `content_block_delta`* / `content_block_stop` triple per
block, then `message_delta` and `message_stop`. Blocks carry types and an index,
and tool inputs arrive as partial JSON fragments that are not valid JSON until
the block closes. So the buffer tracks open blocks rather than accumulating a
string.

It does two things at once: emits OpenAI chunks so the client sees tokens
immediately, and reconstructs the complete Anthropic `Message`. The second is
what keeps the system honest — the reconstructed message goes through exactly the
same translation, usage-extraction, and cost path as a non-streamed one, so
traces and (from week 8) cache values never need a second implementation that can
drift from the first.

Two details that only show up once you build it. OpenAI clients key tool calls by
their index in `tool_calls`, which is *not* the Anthropic block index — one text
block before the first `tool_use` shifts them apart. And a stream cut mid-block
leaves half a JSON object: that is not a tool call with missing fields, it is not
a tool call at all, so it is recorded as truncated rather than parsed
optimistically.

### Two timeout budgets, and the first one is conditional

A response that produces its first token in 400ms and runs 45 seconds is healthy;
one that takes 30 seconds to its first token is not. A single timeout cannot say
that, so first-token and total-duration are separate absolute deadlines. The
first-token deadline is absolute rather than per-event on purpose: `ping` frames
arrive on a healthy idle stream, and resetting the budget on each one would let a
stalled generation ping forever without ever tripping it.

The first-token budget is also *conditioned on the routing decision* — a large
thinking budget legitimately delays the first visible token, so a request with
thinking enabled gets a longer one. That coupling between routing and timeouts is
the sort of thing that only becomes obvious once both exist.

### You cannot cache a response you have not finished streaming

The tee buffers while relaying, which leaves the question of what happens when the
client hangs up at 60%. A partial response is not a cheap cache entry, it is a
wrong one, so it is recorded with `cacheable: false` and the week-8 cache writer
reads that flag before storing anything.

Whether to keep consuming upstream after a disconnect has no clean answer. Prism
abandons by default: closing the connection stops generation, which stops the
meter. `stream_drain_on_disconnect` flips it. The default is worth revisiting in
week 8 — once a semantic cache exists, a completed entry has value that a partial
one does not, and draining buys that value for the price of the remaining tokens.

### Once the first byte is written, the status code is spent

A streamed response commits its status line before the upstream has proven it will
succeed. Prism therefore opens the upstream connection and pulls the first frame
*before* handing the generator to the server, so a 429 or a 529 at stream-open is
still relayed as a real status code rather than degraded into an SSE error frame.
Failures after that point have no channel left except an error frame, and they use
it — a client that is merely cut off cannot tell truncation from completion.

### Raw httpx on the proxy path, the SDK everywhere else

`providers/anthropic_provider.py` speaks raw HTTP. By the time a request reaches
it the body is already a fully-formed Messages API request, and two things the
SDK abstracts away are load-bearing: the `anthropic-ratelimit-*` response headers
that will drive the token bucket in week 10, and byte-level SSE frames for the
week-3 streaming tee. The Anthropic SDK is the right tool wherever Prism is a
*client* rather than a proxy — the eval judge, embedding backfills, the batch
runner — and those will use it.

### Traces cannot fail the request they observe

Trace writes happen in a background task after the response is handed to the
client, and a persistence failure is logged rather than raised.

### Model IDs and prices drift

`registry.py` is the only place either appears. Pin them against
<https://platform.claude.com/docs> rather than trusting the values checked in
here. Introductory pricing windows are modelled explicitly (Sonnet 5 through
2026-08-31) so a cost report run after the window does not silently use stale
rates.

---

### The gate has to be worth keeping switched on

A quality gate is only useful if people do not route around it, so most of the
design here is about not crying wolf.

**It only runs when it can gate something.** The workflow is path-filtered, and
`--auto` exits 0 without a run when the newest prompt version *is* the active
one. A full eval run costs real money; one that fires on a README edit gets
disabled within a week.

**It fails on the interval, not the point estimate.** Covered above — a gate
tripping on every no-op change is the same as no gate.

**A missing measurement is never reported as a pass.** This is the failure mode
the tool is most exposed to, and it was found by running the CLI with no gateway
up: every call errors, every metric reads 0.0 *for both arms*, every delta reads
0.0, and a naive gate calls that "no change" and goes green. So the run carries a
failure rate, `measured()` gates on it, and a broken run exits 2 — a distinct code
from 1, because "the gate could not run" needs a different fix from "quality
regressed". Fork PRs, which get no secrets, fail with an explicit "NOT EVALUATED"
summary for the same reason, and a missing `ANTHROPIC_API_KEY` fails fast before
the run rather than after spending money on 401s.

**The failure explains itself.** The job summary names which metric moved, by how
much, with its interval, and whether the judge behind any judge-derived number was
ever calibrated — an uncalibrated judge is labelled as such rather than presented
as a measurement. A gate that only says "failed" gets overridden.

Two workflows, deliberately separate. `ci.yml` runs lint, types, tests, and
validates the dataset and registry — no API keys, so it runs on forks and every
push, and a lint error is never reported as a quality regression. `eval-gate.yml`
is the required check that costs money.

### Both prompt versions live in the tree at once

The candidate and the active version are both files, so the gate runs them
against the same gateway in the same job. That makes the comparison genuinely
paired and needs no database that survives the run, no checkout of `main`, and no
second environment that might differ.

`manifest.json` names the version serving traffic. Promotion and rollback are the
same operation run in opposite directions — rollback is not a separate emergency
procedure. Versions sort numerically, because sorting `v10` before `v2` as strings
would make the gate compare the wrong pair, silently, and only once a prompt
reached its tenth revision.

The registry hashes the exact bytes of each version, whitespace included. A prompt
edited without a version bump silently invalidates every cached prefix, so the
hash is what gets checked rather than the version number being trusted.

### A bad breakpoint is worse than no breakpoint

Cache writes carry a ~25% surcharge; reads are ~90% off. A breakpoint on content
that changes every request pays the premium every time and never reads back — so
it is strictly worse than not caching at all. The policy therefore refuses to
place a marker in three situations: below the model's minimum cacheable prefix
(the marker is ignored anyway, and the API only gives you four), on the final
message (different on every request by definition), and on the conversation
prefix by default (history only pays back on multi-turn traffic).

The metric that catches a mistake here is the **write/read ratio**, reported per
prompt version by `GET /admin/stats/cache`. Persistently above ~1 means the
breakpoints are sitting on volatile content. That endpoint was built in week 2,
before there was anything to measure, so this work had a number to aim at rather
than a hit rate to admire.

### Breakpoint placement is a security boundary

This is the part worth saying out loud. Prefix cache entries are isolated per
*organization*, not per tenant. A deployment using one API key for all tenants
shares that organization, so the prefix is shared too — and marking a prefix that
contains tenant A's data makes it reachable by tenant B.

So the placer will not accept "this is probably the same for everyone". It takes
an explicit scope per level and refuses to mark anything not proven
tenant-independent, and `policy.py` is what earns that proof: it places a
breakpoint on the system prompt only when the bytes about to be sent are
**byte-identical** to a versioned artifact in the prompt registry. A caller who
claims `assistant@v1` but appends tenant context gets no breakpoint and a recorded
reason, because the claim is now false. Conversation history is never shared. Tool
schemas are shared by default, since they are repo artifacts — but that is a
deployment assumption, so `cache_trust_tools` exists and nothing in the code
pretends it can detect tool descriptions built from customer data.

The defaults are the safe answers, which means the failure mode of forgetting to
configure this is a missed optimisation rather than a leak.

### The token estimator says how wrong it is

Placement needs a token count *before* dispatch, and the ground-truth oracle —
the counting endpoint — costs a round trip, which is unacceptable on the hot path
of a component whose whole job is saving money. So there is a cheap local
estimator plus a calibration harness that measures its error against the oracle
offline.

What the harness reports is the **distribution, both tails**, not an accuracy
figure. The two consumers care about opposite ends: breakpoint placement
over-estimating means marking a block too small to cache (wasted, harmless), while
rate-limit reservation under-estimating means over-committing a bucket and
tripping a 429 (week 10). A mean would hide which is happening. Until someone runs
`calibrate()` against a real key, `EstimatorReport.calibrated` is False and
nothing here claims otherwise.

### The dashboard refuses to show a number it does not have

Every view is server-rendered against the admin API with a tenant-scoped key, so
tenant isolation is enforced once — in the gateway — rather than twice with a
chance of the two disagreeing. The dashboard holds no database credentials.

The design decision worth naming is what happens when a section fails to load.
It renders an explicit *"nothing is being shown for this section — not zero, not
stale"*, never a zero. A dashboard showing $0.00 spend and a healthy circuit
breaker because the gateway is unreachable is worse than no dashboard, and the
whole argument of this project is that an unmeasured number must never look like
a measured one. Empty tables distinguish "no data yet" from "could not load" for
the same reason.

What each view is for:

* **Overview** — spend, prefix hit rate, write/read ratio, and the throughput
  table showing effective ITPM against hit rate. Provider health surfaces
  `429s ignored` deliberately: it counts the quota errors correctly *not* treated
  as provider ill-health.
* **Traces** — every request with its five token classes, plus gateway overhead,
  which is not clamped at zero. Trace detail shows the full translation chain:
  client request, the translated upstream body with its `cache_control` markers,
  the upstream response, and the response translated back.
* **Caching** — both layers side by side, the write/read ratio flagged red above
  1, and the full threshold-versus-false-hit curve with the chosen operating
  point highlighted and its Wilson upper bound beside it.
* **Evaluation** — every metric as an interval. A +2.3% delta whose interval
  spans zero renders as "no change", which is the project's thesis made visible.
  A judge run with no human labels is labelled **uncalibrated** rather than
  quietly reported as a win rate.
* **Prompts & routing** — version hashes with the predicted prefix-cache
  invalidation of a promotion, the derived break-even table, and the router's
  coefficients.

Next 16, incidentally, because the pinned 15.1.6 carried a published CVE and the
only clean fix was the major. Shipping a portfolio project on a known-vulnerable
dependency is not a trade worth making. `npm audit` is clean.

### The prefix cache is a throughput multiplier, not only a cost saver

This is the coupling that only appears once both layers exist. Providers limit
requests, *input* tokens, and *output* tokens independently — three dimensions,
any of which is a 429 — and **cache reads do not count toward the input-token
limit**. So the input budget is really two budgets, and a request expected to hit
the prefix cache reserves almost nothing:

```
effective ITPM = limit / (1 - prefix_hit_rate)
```

At an 80% hit rate a 2M ITPM limit carries roughly **10M** input tokens per
minute. Week 7's cache work turns out to have been throughput work too.

Which creates the real difficulty: the reservation has to *predict* cache
behaviour, and a wrong prediction over-commits the bucket. So every reservation is
reconciled against the `usage` the provider actually reports, and
`under_reservation_rate` — the share of requests that used more than they
reserved — is measured rather than assumed. Those are the requests that can still
trip a 429.

The margin on a reservation comes from the estimator's *upper* error percentile
rather than its mean, because the two directions are not equally bad:
over-reserving wastes headroom, under-reserving trips a limit.

Buckets live in Redis behind a Lua script, because reserving against three
dimensions has to be atomic. A check-then-write from Python would let two
concurrent requests both see room for the last thousand tokens — there is a test
that races twenty requests at a budget of ten and asserts exactly ten win.

Configured limits are only a starting guess. Every response carries the real ones
in `anthropic-ratelimit-*` headers, and `observe()` adopts them, so the gateway
throttles *before* tripping rather than reacting to a 429.

### The 429/529 split finally does something

Week 1 put quota exhaustion and provider overload in separate members of
`ErrorKind`. Ten weeks later that distinction is what the breaker and the failover
chain act on:

| | opens a circuit? | fails over? | why |
|---|---|---|---|
| **429** quota | no | **no** | the quota belongs to the account, not the route — another provider carries the same exhausted budget |
| **529** capacity | **yes** | **yes** | the provider is the problem, and another one probably is not |
| timeout / connection | **yes** | **yes** | whatever is wrong, this route is not serving |
| **400** malformed | no | **no** | malformed everywhere; retrying elsewhere spends a second quota to get the same error |

Conflating the first two is the most common bug in hand-rolled client code, and
it fails in the worst possible direction: a burst of 429s opens every circuit and
the gateway stops serving traffic it could have served. There is a test that fires
a thousand 429s at a breaker and asserts it stays closed.

Closing a circuit needs several consecutive successes, not one — a single success
on a recovering provider is easy to come by, and closing on it puts full load
straight back onto something still fragile. And when every provider fails, the
*real* upstream error is re-raised rather than replaced by a generic 503, so a 529
still reaches the client as a 529 with its `retry-after` intact; the attempt trail
rides along in the body.

### Hedging is a cost decision wearing a latency costume

Firing a duplicate at the p95 mark buys tail latency and pays in tokens: every
hedge that loses is a completion billed and discarded. So it is off by default,
capped by expected cost, and never applied to an expensive call. The rule worth
saying out loud is that a hedged request must never double-reserve rate-limit
budget — the second call draws on the same reservation, or the limiter is wrong by
exactly the hedge rate.

### Two budget caps, and degrade before you reject

One cap forces a choice between surprising someone with a bill and cutting them
off without warning. Two does not: the soft cap warns and keeps serving, the hard
cap acts. At the hard cap the action is configurable, and **degrading to the
cheapest tier usually beats a 402** — the tenant keeps working at a fraction of
the cost and the operator stops absorbing spend.

The running total is a *cache*, not the ledger. `traces` holds every priced token,
so `reconcile()` rebuilds the figure from them; a process that dies between the
upstream call and the update costs accuracy until the next reconcile rather than
corrupting anything. Spend is checked before dispatch and recorded after, so a
tenant can overrun by at most the requests already in flight — closing that window
would mean holding a lock across an upstream call, trading a bounded overrun for
an unbounded latency problem.

Cost attribution is a view over `traces` rather than a second table, and it
divides by **successful** tasks. A tier that fails 40% of the time and retries on
an expensive one is not cheap, and only that denominator shows it.

### The router's threshold is derived, not tuned

Most routers pick a difficulty cutoff by hand and defend it with a plot. There is
no need. Let `p` be the probability the cheap tier produces an acceptable answer,
`c` the cheap call's cost, `e` the expensive one's, and assume a failure is
retried upstream:

```
E[cost | route cheap]     = c + (1 - p) · e
E[cost | route expensive] = e
```

Routing down wins exactly when `c + (1-p)e < e`, i.e. **`p > c/e`**. The threshold
*is* the cost ratio. Measured against the current price list:

| downgrade | cost ratio | route down when |
|---|---|---|
| Haiku 4.5 vs Opus 5 | 0.200 | P(success) > **0.200** |
| Sonnet 5 vs Opus 5 | 0.400 | P(success) > **0.400** |
| Haiku 4.5 vs Sonnet 5 | 0.500 | P(success) > **0.500** |

Four wasted Haiku calls still cost less than one avoided Opus call, so routing
down to Haiku is right far more often than instinct suggests. And the numbers move
with the price list — Haiku-vs-Sonnet sits at 0.5 right now only because Sonnet is
on introductory pricing, which is exactly the kind of thing a hand-tuned constant
would get silently wrong.

Two refinements: a verifier is paid on *every* request, so verify-then-escalate
raises the bar to `(c+v)/e`; and without escalation a bad cheap answer is simply
delivered, so money always favours routing down and a **quality floor** has to
bind instead. Both modes are implemented and the difference is explicit.

Because the rule compares a probability against a cost ratio, the classifier has
to be **calibrated**, not merely well-ranked. Brier score is reported next to AUC
for that reason.

### Why logistic regression, and what that buys

A few hundred rows, a few dozen features, one binary question. Logistic regression
trains in under a second, needs no GPU, and — the part that matters — has readable
coefficients. From the demo run:

```
lookup_markers     +0.2492      reasoning_markers  -0.2492
n_question_marks   +0.2492      avg_word_length    -0.3138
                                query_chars        -0.2717
```

"Why was this routed up?" has a one-sentence answer. Fine-tuning a model for this
would cost orders of magnitude more, be unexplainable, and be worse — the signal
lives in request shape, not in language modelling. Reaching for the simplest
sufficient tool is the point, and being able to say why is most of the value.

The embedding is projected from 384 dimensions down to a handful before it reaches
the model. That is a deliberate bias toward underfitting: a router that memorises
its training set reports a quality-retention figure that is a lie.

### Two axes, and the bug that proved the second one was real

Tier is discrete; effort is continuous. Thinking tokens bill as output, so effort
scales the output price rather than switching price lists — which means trimming
effort on an expensive tier can beat moving to a cheap tier at full effort.

The first implementation banded effort on `p_success`. A test caught that this
makes the second axis **dead**: `p_success` is also what picks the tier, so
everything reaching Opus has `p < 0.2` by construction and the band table handed
out `"high"` every time. Effort now bands differently by rung — at the top, *how*
hard (medium → high → xhigh); below it, by what margin the break-even was cleared.

### Contamination is guarded before fitting, not reviewed after

The router trains on eval outcomes, so training and scoring it on the same
examples turns quality retention into memorisation. `assert_disjoint` runs inside
`build()`, before any fitting. The subtler trap is that every label depends on a
scoring threshold — "the cheap tier succeeded" means it scored 1.0 on its
strictest metric — so `success_threshold` is an explicit recorded parameter.
Lowering it makes the router look better and the product worse.

Two failure modes are refused outright: training data with only one outcome class
(a boundary cannot be learned from examples that all went the same way), and a
trained router with AUC below 0.60 (`train_router.py` exits non-zero — a coin flip
wearing a coefficient vector is worse than picking one tier, because it adds
latency and a moving part to reach the same answer).

An untrained or missing model routes **up**, to the top of the ladder. Falling
back to the cheap tier would silently degrade quality the moment a file went
missing.

### The semantic cache can be wrong, and everything follows from that

A prefix-cache miss costs money. A semantic false hit returns someone else's
answer to a real user. Those are not comparable failures, so:

**The threshold is an operating point on an ROC curve, not a tuned constant.**
`calibrate_cache.py` sweeps thresholds against hand-labelled pairs and picks the
best recall subject to a false-hit *ceiling* — applied to the **upper bound** of
the Wilson interval, not the point estimate. On a small labelled set the point
estimate is often 0.0, and treating that as proof of safety is how a cache ships
with an unmeasured error rate. When nothing qualifies, the tool says so and exits
non-zero; "this cache should stay off" is a real answer.

**The labelled set is mostly hard negatives.** "What is the capital of France?"
against "…of Germany?" — one token apart, different correct answer. `HTTP` against
`HTTPS`. "Convert 100 USD to EUR" against "…EUR to USD". Without pairs like these
the curve is easy, the chosen threshold is worthless, and the cache looks safe
right up until it isn't.

**Scope is a partition, not a filter.** Tenant, model, temperature, system-prompt
hash and tool-schema hash fold into one key, and a nearest-neighbour search never
crosses it. Filtering after retrieval would mean the leak exists in the query path
and is prevented only by a later `if`. The tenant case has its own test, and it is
the one property no threshold can provide: two tenants asking the identical
question have similarity 1.0.

**The cache refuses to run on a stub embedder.** `HashingEmbedder` reports
`is_production_grade = False` and the constructor raises unless a test explicitly
opts in. Its measured AUC is 0.24 — worse than chance, because character trigrams
make near-identical strings with *different* meanings the closest pairs. A cache
running on it would serve confident nonsense.

**Misses record how close they came.** Production traffic then becomes the data
for re-tuning the operating point, instead of needing a fresh labelling exercise
every time the prompt or model changes.

### Why the embedding model runs locally

Calling a paid embedding API to decide whether you can avoid a paid completion API
defeats the exercise. At ~$0.02/M for a hosted embedding against $5/M for Opus
input, the hosted option eats a visible slice of the saving it exists to produce,
and adds a network round trip to the hot path of *every* request, hit or miss.
`bge-small-en-v1.5` is ~33M parameters on CPU, so the saving is bounded by
electricity. It is an optional extra (`pip install -e ".[embeddings]"`) because
torch is large and a deployment with layer 2 off should never load it.

### The eval harness came before the cache and the router

Both need it as ground truth. The cache's operating point is a precision/recall
tradeoff that can only be chosen against measured quality, and the router trains
on labelled outcomes this harness produces. Building either first would mean
tuning them against a number invented afterwards to justify the result — which is
how the eval story becomes an afterthought, and the failure mode this project is
organised to avoid.

### Report intervals, not point estimates

On a 200-example set, whether +3% is real depends on the *disagreement pattern*,
not the margin. An arm that wins 13 examples and loses 7 nets +3% with an interval
spanning zero — nothing has been shown. An arm that wins 6 and loses none nets the
same +3% and is detectable. Both cases are pinned as tests.

Comparisons are **paired**: both arms answer the same examples, so bootstrapping
resamples examples rather than arms. Discarding the pairing throws away the fact
that a hard example is hard for both and hides real effects behind example-level
variance. The bootstrap seed is fixed, because an interval that moves between runs
of identical data invites re-rolling until the answer is the one you wanted.

The gate fails only when the *entire* interval sits below tolerance. A gate that
fired on the point estimate would fail roughly half of all no-op changes and be
switched off within a week.

### The judge is an instrument, and instruments get calibrated

Three biases are handled structurally rather than hoped away:

* **Position bias** — judges favour whatever they read first. Every comparison
  runs in both orders, and a pair only counts as a win if the *same response*
  wins twice. Flip across orders and it is recorded as `inconsistent`, which is
  information: a rising inconsistency rate means the judge cannot tell the arms
  apart, and the honest reading is "no measurable difference", not a coin flip.
  The rate is reported on its own.
* **Self-preference** — models score their own output higher. Prism serves Claude
  on every tier, so `assert_different_family()` refuses a Claude judge outright.
  Staying single-vendor on the serving side makes this *checkable* rather than a
  rule someone has to remember.
* **Verbosity bias** — the rubric names it, and the judge returns a verdict token
  rather than an essay, which also makes parsing deterministic.

None of that makes the judge correct. Cohen's kappa against ~100 human labels is
what says whether it agrees with people better than chance. Raw agreement will not
do: two raters who both say "tie" most of the time agree 80% of the time and score
κ = 0.41 — "moderate". Every judge-derived number is reported with its kappa
attached, and a run whose judge falls below substantial agreement prints a warning
rather than a clean result.

### Splits exist to stop the router grading its own homework

The week-9 router trains on outcomes this harness produces. Train and score it on
the same examples and its reported quality retention is memorisation. `split` is
required on every example, `load()` refuses a file whose ids repeat across splits,
and `assert_disjoint()` is called before any training run.

### Cost per successful task, not cost per token

A cheap arm that fails 40% of the time and triggers a retry on the expensive one
costs more than going straight to the expensive one. `ArmScores` reports both, and
a test pins the case where the cheaper arm wins on total cost and loses on cost per
success. Failed calls score zero rather than being dropped — dropping them would
let an arm improve its average by erroring on the examples it finds hard.

## Layout

```
src/prism/
  config.py         typed settings
  registry.py       model capabilities + pricing — the only place either lives
  cost.py           five-token-class cost model
  api/              ingress routes, admin routes, auth dependency
  schemas/          OpenAI wire types; error taxonomy
  translation/      OpenAI <-> Anthropic, both directions; stream.py is the
                    block-structured reassembler
  providers/        provider adapters (Anthropic today; the interface is for week 10)
  db/               SQLAlchemy models, engine, repository
  tracing/          trace assembly and background persistence
  eval/             golden dataset, metrics, judge, calibration, bootstrap stats
  prompts.py        versioned prompt registry; cache-key and gate input
  tokens.py         local estimator + calibration against the counting endpoint
  caching/          both layers: breakpoint policy, scope keys, embeddings,
                    pgvector store, threshold calibration, three-config replay
  routing/          break-even economics, features, classifier, two-axis policy
  reliability/      three-dimensional limiter, breakers, failover, hedging
  governance/       per-tenant budgets; cost attribution is a SQL view
dashboard/          Next.js reader of the admin API; holds no credentials
migrations/         idempotent SQL, applied on first boot and by scripts/migrate.py
datasets/golden/    the golden set, versioned next to the code
datasets/cache_pairs.jsonl  labelled query pairs, mostly hard negatives
prompts/            prompt versions + manifest.json naming the active one
.github/workflows/  ci.yml (free, always) and eval-gate.yml (costs money)
```

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible, streaming and non-streaming; model `prism-auto` hands tier selection to the router |
| GET | `/v1/models` | the ladder, with capability flags |
| GET | `/admin/traces` | tenant-scoped trace list |
| GET | `/admin/traces/{id}` | full request/response inspection |
| GET | `/admin/stats/cache` | hit rate and write/read ratio by prompt version |
| GET | `/healthz` | |

Every completion response carries `x-prism-model` (what actually ran),
`x-prism-cost-usd`, `x-prism-cache-read-tokens`, `x-prism-cache-write-tokens`,
`x-prism-upstream-latency-ms`, and `x-prism-gateway-overhead-ms`.

## Build order

1. **Weeks 1–2** — proxy skeleton, translation, trace persistence ✅
2. **Week 3** — streaming: block-structured event reassembly, tee-and-buffer,
   first-token vs total timeout split, client-disconnect handling ✅
3. **Weeks 4–5** — eval harness *before* the cache and the router, because both
   need it as ground truth ✅
4. **Week 6** — CI regression gate ✅
5. **Weeks 7–8** — prefix breakpoint policy and measurement, then the semantic
   layer with its labelled pair set and ROC curve ✅
6. **Week 9** — the two-axis router ✅
7. **Weeks 10–11** — rate limiting, circuit breakers, failover, budgets ✅
8. **Week 12** — dashboard ✅
