# Prism

A Claude-native LLM gateway: OpenAI-compatible ingress, Anthropic Messages API
egress, layered caching, a two-axis adaptive router, and a CI-gated evaluation
harness.

**Status: week 7 of 12.** The proxy, protocol translation (streaming and not),
cost model, trace persistence, evaluation harness, CI quality gate, and the
provider prefix-cache layer are in. Everything else is scaffolding with a date
on it — see [Build order](#build-order).

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
| Semantic cache (pgvector/HNSW) + ROC calibration | week 8 |
| Two-axis router | week 9 |
| Rate limiting, circuit breakers, budgets | weeks 10–11 |
| Dashboard | week 12 |

---

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
  caching/          breakpoint placement policy and the scope decision behind it
migrations/         idempotent SQL, applied on first boot and by scripts/migrate.py
datasets/golden/    the golden set, versioned next to the code
prompts/            prompt versions + manifest.json naming the active one
.github/workflows/  ci.yml (free, always) and eval-gate.yml (costs money)
```

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible, streaming and non-streaming |
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
5. **Weeks 7–8** — prefix breakpoint policy and measurement ✅, then the
   semantic layer with its labelled pair set and ROC curve
6. **Week 9** — the two-axis router
7. **Weeks 10–11** — rate limiting, circuit breakers, failover, budgets
8. **Week 12** — dashboard
