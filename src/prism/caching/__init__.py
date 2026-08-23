"""Caching layers.

Three of them, in increasing order of risk:

1. **Provider prefix cache** (`breakpoints.py`, `policy.py`) — exact-prefix,
   enforced by the provider, and *cannot return a wrong answer*. Only ever costs
   money when placed badly.
2. **Semantic cache** (week 8) — fuzzy nearest-neighbour lookup, which *can*
   return a wrong answer to a real user. Needs a calibrated threshold and a
   labelled pair set before it is allowed anywhere near traffic.
3. **Exact-match response cache** (week 8) — Redis, keyed on the whole request.

The ordering is the argument. Layer 1 captures the easy wins at zero correctness
risk, which is why it was built first and why layer 2's contribution has to be
measured as the *incremental* saving over it rather than as an unconditional hit
rate.
"""
