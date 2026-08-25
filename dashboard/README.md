# Prism dashboard

A read-only Next.js client of the gateway's admin API.

```bash
npm install
cp .env.example .env.local   # point PRISM_URL at the gateway, add a tenant key
npm run dev                  # http://localhost:3001
```

## Two things about the design

**It holds no database credentials.** Every view is server-rendered against
`/admin/*` with a tenant-scoped API key, so tenant isolation is enforced once —
in the gateway — rather than twice with a chance of the two disagreeing.

**It will not show a number it does not have.** When a section fails to load it
says so explicitly, and says what is *not* being shown. It never renders a zero.
A dashboard reporting $0.00 spend and a healthy circuit breaker because the
gateway is unreachable is worse than no dashboard; empty tables distinguish "no
data yet" from "could not load" for the same reason.

Nothing is cached (`cache: "no-store"`). A cached page showing a closed circuit
breaker that opened two minutes ago is exactly the failure this is here to catch.
