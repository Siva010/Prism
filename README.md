# Prism

A Claude-native LLM gateway: OpenAI-compatible ingress, Anthropic Messages API
egress, layered caching, a two-axis adaptive router, and a CI-gated evaluation
harness.

**Status: week 3 of 12.** The proxy skeleton, protocol translation (streaming and
not), cost model, and trace persistence are in. Everything else is scaffolding
with a date on it — see [Build order](#build-order).

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
    model="gpt-4o",                       # mapped onto the Claude ladder
    messages=[{"role": "user", "content": "capital of France?"}],
)
```

Tests, lint, and types:

```bash
pytest && ruff check . && mypy src/prism
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
| Eval harness, judge calibration, CI gate | weeks 4–6 |
| Prefix-breakpoint policy, semantic cache | weeks 7–8 |
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
migrations/         idempotent SQL, applied on first boot and by scripts/migrate.py
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
   need it as ground truth
4. **Week 6** — CI regression gate
5. **Weeks 7–8** — prefix breakpoint policy and measurement, then the semantic
   layer with its labelled pair set and ROC curve
6. **Week 9** — the two-axis router
7. **Weeks 10–11** — rate limiting, circuit breakers, failover, budgets
8. **Week 12** — dashboard
