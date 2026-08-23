# Prompt registry

`prompts/<name>/<version>.md`, with `manifest.json` naming the version currently
serving traffic.

Both the active version and any candidate live in the tree at the same time.
That is what lets the CI gate run a paired comparison inside one job, with no
git-checkout gymnastics and no dependency on a database that survives the run.

## Shipping a change

1. Add `prompts/<name>/v<N+1>.md`. Leave `manifest.json` alone.
2. Open a PR. The eval gate sees a version newer than the active one and scores
   the pair; it fails the build if quality drops beyond the confidence interval.
3. When the gate passes, promote it — this is also the rollback, in reverse:

   ```bash
   python scripts/promote.py assistant v2
   ```

## Why versions are files rather than rows

A prompt version is a cache-key input and a prefix-cache breakpoint boundary, so
it has to be reviewable in a diff alongside the code that uses it. Editing a
prompt's text without bumping its version silently invalidates every cached
prefix, which is why the registry hashes the exact bytes and the gate compares
hashes rather than trusting the version number.
