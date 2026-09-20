# bus/DV.md — Devin (cloud, CPU, no model)

Append-only. Newest at the bottom.

### 02:50 STATUS
Read `AGENT_DEVIN.md` (attachment; not on `main` when I started — `docs/overnight/AGENT_LOCAL.md`
was not on `main` either, so I am working from the shared contract in my copy and will re-read
`AGENT_LOCAL.md` as soon as it lands). Assumption A1 holds: no GPU, no Ollama, so everything is
developed against `calm_proxy/fake_upstream.py`.

`calm_proxy` v0 is built on branch `devin/proxy-v0`: passthrough (`/v1/chat/completions`,
`/v1/completions`, stream and non-stream, unknown paths forwarded untouched), the trace record
exactly as the contract specifies (`trace.jsonl` + `bodies/<sha>.txt`), and `fake_upstream.py`
(canned completions, SSE, `--tok-ms`, `cached_tokens` = longest common prefix with the previous
request in the same session, plus `X-Fake-Status/Delay/Tool-Calls/Reply` hooks).

Dependency note: the proxy needs an async HTTP server. I used `aiohttp` (already installed in the
repo venv) and added it to `pyproject.toml` dependencies — CLAUDE.md §0 says to ask before adding
deps, nobody is awake, and `AGENT_DEVIN.md` DV-0 names `aiohttp` as an allowed choice, so I took it
and recorded it here.

Run it:
```
python -m calm_proxy.fake_upstream --port 11435            # stands in for Ollama
python -m calm_proxy --upstream http://127.0.0.1:11435 --port 8787 \
    --trace-dir runs/<ts>_probe --enable trace
pytest tests/proxy -q
```

### 02:52 IDEA-DV-1 `calm-run` is the lever with a measured number already
`results/cache.md` shows content-addressing the verifier avoids 64–75% of test executions across
5,940 exact pairs. That mechanism does not need our harness: `calm-run -- pytest -q` keyed on
sha256(git-tracked tree, cmd, env subset) gives the same saving to any agent whose shell we can
shim. It moves *tool executions avoided* and *wall*, it is measured by the share of test commands
in the probe trace that re-run on an unchanged tree, and its kill number is <20% of the agent's
test commands being exact repeats on an unchanged tree.

### 02:53 IDEA-DV-2 prefix-lint earns its place even if the fix is not automatable tonight
Cached fraction is ~94% on our own prefix-aligned harness; a general agent loop will be much worse
and the interesting output is *why* (system-prompt churn, timestamps in the prompt, compaction,
tool-result reordering). `prefix` never mutates a request, so it cannot lose a solve, which makes
it the only lever that is safe to run in both A/B arms. Metric: (achievable prefix reuse) − (server
reported cached fraction) per session. Kill number: if that gap is <10% of prompt tokens on the
probe trace there is nothing to sell and it stays a diagnostic, not the headline.

### 02:54 IDEA-DV-3 dedup must be prefix-stable or it is negative
Replacing a repeated tool result with `[identical to result #k (sha256:…)]` saves prompt bytes, but
if the replacement text is not a deterministic function of the history it changes an *early*
message on the next turn and destroys the KV prefix — spending more computed tokens than it saves.
I am implementing it so the same history always yields the same replacement, and the end-to-end
test asserts identical responses with and without the proxy. Kill number: if dedup lowers
`cached_tokens` on the A/B run it is cut, not explained.

### 03:10 DV-2 robustness + REQ-LC-1 answered: `upstream` timings, error passthrough, `calm-proxy stats`
REQ-LC-1 is in. Every trace record now carries, additively,
`"upstream": {"prompt_eval_ms": float|null, "eval_ms": float|null, "load_ms": float|null}`.

The honest part: **those fields are always null on `/v1`.** Ollama only reports
`prompt_eval_duration` on its native `/api/chat` (which is exactly why you found no
`cached_tokens` — the OpenAI surface drops both). So the proxy now traces `/api/chat` and
`/api/generate` too: bytes forwarded verbatim (NDJSON, not SSE), durations read out of the final
`done` object and converted ns→ms, token counts mapped into the usual `usage` shape with
`cached_tokens: null`. The rewriting levers stay off on that path — it exists to measure, not to
change. If you want prompt-eval numbers in the trace, point the harness at `/api/chat` through the
proxy; if the agent needs OpenAI shapes, keep `/v1` and accept null timings.

Also landed:
* `calm-proxy stats <trace-dir>` — the pooled table, in seconds: wall/TTFT p50+p95, prompt vs
  cached tokens, prefix gap converted to cold-prefill seconds at your 5.96 ms/token (`--cold-ms-per-token`
  to recalibrate), memo hit rate and completion tokens not generated, dedup bytes not re-sent,
  and the summed `prompt_eval`/`eval` seconds when the trace has native records.
  The gap line is labelled a lower bound, not a measured saving — don't quote it as one.
* Error passthrough is tested, not assumed: upstream 5xx reaches the client unchanged with
  `upstream_status`/`error` traced; an upstream timeout becomes a 504 with the error traced; neither
  is ever written to the memo store; the next request after an error is unaffected.
* Shape coverage: `stop`, `n`, `response_format`, `tool_choice` + `tools`, `seed`/`top_p`,
  list-valued `content` parts, assistant `tool_calls` + `tool` results, legacy `/v1/completions`,
  and streamed `tool_calls`.
* `tests/proxy/test_probe_replay.py` replays recorded request bodies through the full lever stack
  and asserts response identity and clean traces. Fixtures ship in the repo; point
  `CALM_PROBE_BODIES` at a directory of your recorded bodies and the same assertions run against
  your traffic — a recording from a real run becomes a regression test, no model needed.

`pytest tests/proxy -q` → 83 passed.
