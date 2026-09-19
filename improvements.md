# Improvements: Making `calm_coder` Faster

This document identifies niches to attack to turn `calm_coder` into a harness that makes coding genuinely *faster*. Token usage and accuracy are explicitly deprioritized in favor of wall-clock latency.

## What this project is

`calm_coder` is a multi-agent class-generation harness built on the CALM principle (Consistency As Logical Monotonicity). N agents write every method of a class concurrently; nothing overwrites anything. Definitions land in a grow-only, content-addressed store (`add_def`, `add_outcome`, both idempotent). Methods are keyed by the hash of their canonical form, and calls to other declared methods are late-bound so any fill composes with any other.

Pipeline (in `calm_coder/agents/`):
- Phase 0 (`fill.py` `generate_fills`): generate N fills per slot concurrently, ingested as they return.
- Phase 1 (`scheduler.py` `stub_test`): run each fill's own slot test with everything else stubbed.
- Phase 2 (`scheduler.py` `search`): composition search over a ranked frontier until `∃ verified composition`.

The existing latency notion is TTFV (time-to-first-verified), hypothesis H3 in `CLAUDE.md`, which the spec itself flags as the shakiest bet. That is the opening.

## Niches to attack (ordered by leverage)

### 1. Batch/parallelize test execution in Phase 2
`search` in `calm_coder/agents/scheduler.py` runs full-module tests one composition at a time via `run_full`, draining the frontier sequentially. Frontier compositions are independent facts, so run the top-K concurrently and take the first that verifies. Pure latency win that spends more CPU; confluence guarantees racing changes nothing derivable. Also add per-test-class caching keyed by (test_id, reachable bindings) instead of only per-composition (`self._full[comp.id]`).

### 2. Speculative test execution during generation
`generate_fills` gathers all samples before Phase 1 begins, but fills are ingested the moment they return and the store is monotone. Start Phase 1 stub-tests (and speculative Phase 2 compositions) while later samples are still decoding. Directly attacks TTFV.

### 3. Latency-first `calm solve` dev-tool path
`calm_coder/cli.py` runs strictly sequential phases (gather all fills → phase1 → search). For a developer-facing tool, latency-to-first-working-class is the whole UX. Stream partial results, start testing before all N fills land, and early-exit the moment any composition verifies (the exit is already `∃`-based).

### 4. Adaptive N and adaptive frontier width
N is a fixed per-slot budget; `max_comps` defaults to 64. A "fast mode" spends unequally: few fills for slots that pass their stub test immediately, more samples into repeatedly-failing implicated slots (from `_callee_closure`).

### 5. Prefix-cache exploitation for latency
Slot prompts share a byte-identical prefix with the slot name last. The spec frames this as a cost lever; on a real server (vLLM `--enable-prefix-caching`) it is also a latency lever cutting TTFT for parallel slot requests. Re-pitch and measure prefill/decode latency instead of tokens.

### 6. Persistent cross-run outcome cache
Definition identity is a content hash and derivations are pure, so (composition, test) outcomes are stable forever. A persistent cross-run cache keyed by def hashes + test source hash skips test execution when re-solving a class or one sharing helpers. The store is already replay-safe and event-log-based.

## Status

Implemented in `calm solve` (default path) and, where reproducibility allows, in the experiment:

| Niche | Where | Note |
|---|---|---|
| #1 parallel Phase 2 | `Scheduler(width=…)`, `search` | Frontier batches of `width` race; the first verified in batch order is the exit. Expansion still happens per composition, so the frontier is the same set. `--test-width` in the experiment; `SearchResult.test_wall_ms` measures the win. |
| #1 per-class caching | `Scheduler.class_key`, `_reused` | Keyed by the test-module hash, the test class, and the fills it can reach (`test_slot_deps` closed over `_callee_closure`) plus which reachable slots are stubbed. A new composition re-runs only the classes that can see the swapped slot. |
| #2 speculative testing | `generate_fills(on_emission=…)`, `cli._pipelined` | Stub tests start per fill as it lands; a composition search runs as soon as every slot has one fill, then again on each new fill. |
| #3 latency-first CLI | `cli._pipelined` (default; `--sequential` opts out) | Generation is cancelled at the `∃` exit; in-flight tests are drained so their outcomes still land. |
| #4 adaptive N | `generate_fills(skip_slot=…)` + `Scheduler.has_stub_pass` | Waves: a slot with a stub-passing fill draws no further samples. Scheduling only — it changes which facts exist, never which are derivable, so the experiment never uses it. |
| #6 cross-run cache | `calm_coder/runner/cache.py`, `--cache` | Append-only JSONL keyed by the same content hash. Reused outcomes carry `reused:` in their detail; the log never claims an execution that didn't happen. |

Not done: #5 (prefix-cache latency) needs a server that reports prefill/decode separately — nothing to
implement here beyond metrics, and the laptop's Ollama doesn't expose them.

Both reuse mechanisms only ever *add* outcomes the store would have derived anyway, so I1–I3 hold: the
store stays grow-only, the keys are content hashes, and derivations still read sets.

## Honest caveat

The decision log in `CLAUDE.md` measured the laptop reality: qwen2.5-coder:7b ~20 tok/s single-stream, ~25 tok/s aggregate, and 8 parallel sequences pushed the model off the GPU. On the laptop, generation is the wall and test parallelism competes for the same cores. Niches #2 and #3 (pipelining + early exit) are best for the laptop demo; #1, #5, #6 dominate on rented GPU compute.
