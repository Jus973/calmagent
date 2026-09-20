# DECISIONS — what ships

> Owner: CF. **Written by LC under the 03:55 fallback rule** (contract, Clock row T+1:00): CF's
> decision was not on `main`, and as of 03:20 neither `bus/CF.md`, `bus/DV.md`, `calm_proxy/` nor
> `analysis/proxy_report.py` exists on `main`. If CF lands a `DECISIONS.md`, **theirs wins** and
> this file should be replaced wholesale, not merged. Everything below is reversible: no lever has
> been deleted, only ranked.

## The decision

**Headline lever: I-4 `dedup`.** It is the only candidate of I-2…I-5 whose kill number is not hit
on the probe trace, and it is the only one aimed at the thing the trace says is actually expensive.

**Also ships:** I-1 `trace` (it is the instrument, and it is what produced every number here) and
I-5 `calm-run` (cheap, independently evidenced, and honestly caveated below).

**Does not ship as a lever:** I-2 `prefix-lint` ships as a *diagnostic* only — it prints a clean
bill of health on this agent, which is a result, not a feature. I-3 `memo` is dropped. I-6
`affinity` stays cut.

## Why, from the probe trace (42 requests, `bench_agent/probe_headroom.py`)

| Lever | Measured headroom | Kill number | Verdict |
|---|---|---|---|
| I-2 prefix-lint | **0.0 s** (churn tokens: 0; all 40 non-first requests diverge `appended_only`) | **HIT** | diagnostic only |
| I-3 memo | **0.0 s** (0 duplicate request keys of 42) | **HIT** | drop |
| I-4 dedup | **102 s** of uncached duplicate bytes; prompts 38–44% smaller on replay | not hit | **headline** |
| I-5 `calm-run` | 2 reruns of 5 test commands | not hit, but ~40 ms on this bench | ships, caveated |

Supporting measurements, all from run directories, none typed by hand:

- A prefix-cache hit is worth **100.9×** on prompt evaluation (25,711 ms → 253 ms on a 4,317-token
  prompt) — `runs/20260920T063803Z_cache_probe/`.
- This agent already gets **94.8%** of that available saving (measured prefill 283.3 s, against
  185.8 s for a perfect cache and 2,072.7 s for none). There is no cache saving left to win, which
  is exactly why I-2 cannot be the headline.
- Cost per *new* token rises with context length on rows doing identical work: **3.99 ms at a 1k
  prompt, 9.54 at 8k, 17.40 at 17k.** No prefix cache addresses this; a shorter context does.
- The agent sent **347,775 prompt tokens to get 4,514 completion tokens** — 77:1.

## The metric

The contract's headline metric was "prompt tokens actually computed (prompt − cached)". **It is not
available and it would be ~0 anyway.** Ollama 0.21.2 reports no `cached_tokens` on `/v1`, and
`prompt_eval_count` does not shrink on a cache hit, so the quantity cannot be read from the server;
and at 94.8% capture there is almost nothing in it. Replaced by:

1. **Upstream wall clock**, decomposed into decode and prefill. Primary.
2. **Prompt tokens sent** and **context size per turn**. This is what dedup moves directly.
3. **Solve rate**, with a Wilson interval per arm and an exact McNemar on the paired tasks. Per the
   contract this must include 0 or favour `on`; if `on` loses solves, dedup is cut, not explained.

`prefill_s` is a derived decomposition (upstream wall minus completion tokens × a measured decode
constant), because Ollama does not split the two on the OpenAI-compatible endpoint and the agent
does not stream. Every document that prints it must say so. `bench_agent/quick_report.py` does.

## Proxy flags for the A/B

```
off:  (trace only)
on:   --dedup --dedup-min-bytes 200
```

`off` is **not** "straight to Ollama". Both arms run through the same proxy, `off` with tracing
only. A control arm bypassing the proxy would have no trace at all — Ollama reports no usage on a
streamed request and no cache figures on any request — so there would be nothing to compare.
Tracing relays bytes and writes a line; it does not touch the request. The single exception is
documented and identical in both arms: on a streamed request the proxy sets
`stream_options.include_usage` so the server returns a token count.

## What the README must not claim

- Not "dedup saves 620 s of prefill". Most removed bytes were already cached at 0.059 ms/token.
  The defensible prefill figure is 102 s; the rest of the case for dedup is the context-length
  curve, and the A/B is what tests it. If the A/B comes back flat on wall clock, dedup is a
  context-window lever, not a speed lever, and the README says that.
- Not "we made the cache work better". This agent's cache already works. What the proxy did was
  **prove** it, and the same instrument shows a 100.9× cliff waiting for any agent whose prompt
  is not append-only.
- Not `results/cache.md`'s 64–75% as a wall-clock claim. That was a count of executions avoided; a
  ClassEval test module runs in ~20 ms, so on this bench it converts to ~40 ms. A repo with a 30 s
  suite is where I-5 pays, and this bench cannot show it.

## Open, for CF if they arrive

- Whether `prefix-lint` shipping as a diagnostic is enough for the README's story, or whether the
  demo should include a deliberately badly-behaved agent (a timestamp in the system prompt) so the
  100.9× cliff is shown rather than described. The cache probe already measures that case
  (`a_ts_churn_1/2`: 28.6 s and 28.7 s against 0.49 s warm), so it would be a presentation change,
  not new measurement.
