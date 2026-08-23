# Prism

A Claude-native LLM gateway: OpenAI-compatible ingress, Anthropic Messages API
egress, layered caching, a two-axis adaptive router, and a CI-gated evaluation
harness.

**Status: week 2 of 12.** The proxy skeleton, protocol translation, cost model,
and trace persistence are in. Everything else is scaffolding with a date on it —
see [Build order](#build-order).

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
| OpenAI `/v1/chat/completions` ingress, non-streaming | done |
| OpenAI → Anthropic request translation | done |
| Anthropic → OpenAI response translation | done |
| Five-token-class cost model | done |
| Trace persistence (Postgres) + admin read API | done |
| Hashed per-tenant API keys | done |
| Error taxonomy with the 429/529 split | done |
| SSE streaming | week 3 |
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

### Streaming returns 501, deliberately

`stream: true` is refused rather than quietly served as one buffered blob. A
client that asked for tokens as they arrive and got a single response at the end
has been given a different product — and hiding that would make week 3 look done
before it is.

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
  translation/      OpenAI <-> Anthropic, both directions
  providers/        provider adapters (Anthropic today; the interface is for week 10)
  db/               SQLAlchemy models, engine, repository
  tracing/          trace assembly and background persistence
migrations/         idempotent SQL, applied on first boot and by scripts/migrate.py
```

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible; `stream: true` returns 501 until week 3 |
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
   first-token vs total timeout split, client-disconnect handling
3. **Weeks 4–5** — eval harness *before* the cache and the router, because both
   need it as ground truth
4. **Week 6** — CI regression gate
5. **Weeks 7–8** — prefix breakpoint policy and measurement, then the semantic
   layer with its labelled pair set and ROC curve
6. **Week 9** — the two-axis router
7. **Weeks 10–11** — rate limiting, circuit breakers, failover, budgets
8. **Week 12** — dashboard
