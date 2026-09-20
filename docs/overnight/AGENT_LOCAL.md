# Overnight instructions for Claude Code (local, the Mac with Ollama) — "LC"

Read this whole file, then `docs/overnight/AGENT_DEVIN.md`'s non-shared part (10 lines) so you know what Devin is building. The shared contract below is byte-identical in both files. Re-read it every hour; long sessions lose detail.

You run **unattended until ~08:00 ET**. Nobody answers questions. Every rule you need to make a decision is here. If you hit an action that needs permission, try it once, record it under **"USER: read this first"** in `STATUS.md`, and move to work that does not need it. Do not sit on a prompt.

## Your role in one paragraph

You are the only agent that can generate tokens, so you own every **measurement**. Devin builds the proxy package; Claude-cloud (CF) analyses traces and writes the report. You build the agent-loop bench, run one real agent through the proxy, produce the probe trace that decides the headline, then run the A/B and commit the run directories. The product is judged on your numbers, so your first priority is a trace by 03:30 and your last is A/B directories on `main` by 06:30.

## Step 0 — first 15 minutes (do these in order, do not skip to the fun part)

1. `git pull --rebase origin main`. Confirm `docs/overnight/`, `bus/` exist. Create `bus/LC.md` with a `STATUS` entry: Ollama up? model loaded? `ollama ps`. Push.
2. Rewrite `STATUS.md` (yours) to the new shape: **"USER: read this first"** (3 bullets, empty for now), then **Research phase result** (two lines: null on solve rate, links to `results/final.md`), then **Pivot** (link to `docs/overnight/`), then a running log. The old content moves under a `## Archive (research phase)` heading; do not delete it.
3. Check the laptop will survive the night: `caffeinate -dims &` (or confirm `pmset -g` shows sleep disabled), Ollama serving on 11434, `curl localhost:11434/v1/models`. Record in the bus. If `caffeinate` is not allowed, note it under "USER: read this first" and continue.
4. Post `IDEA-LC-1` and `IDEA-LC-2` in `bus/LC.md` before 03:15 (see the seed-ideas table in the contract; you may sharpen one of those or add new ones). Two paragraphs each, with a kill number.

## Step 1 — the agent loop (target: one traced task by 03:30)

Build `bench_agent/`:

- `bench_agent/make_tasks.py`: from `calm_coder.bench.classeval` (read only), write `bench_agent/tasks/<ClassEval_id>/` with the skeleton as `solution.py`, the tests as `tests/test_solution.py` (unittest, importable), and a `TASK.md`: "Implement the class in solution.py so that `python -m pytest -q` passes. Do not edit tests." Pick 10 tasks with seed 0: 8 from those the 7B model solved in `runs/20260919T163449Z_main` (any arm), 2 it did not. Write the list to `bench_agent/tasks/TASKS.json`. Commit the tasks.
- `bench_agent/run_agent.py --agent {mini-swe,aider,calm} --task <id> --base-url <url> --out runs/<ts>_<name>/`: copies the task to a temp dir, runs the agent with a hard wall-clock cap (**8 minutes per task**, then kill), then runs the hidden tests itself and writes `result.json` (solved, wall, agent exit reason). The agent's model calls go through `--base-url`.
  - Try `mini-swe-agent` first: `pip install mini-swe-agent`; model `ollama/qwen2.5-coder:7b`, `api_base` = the proxy. 20 minutes max to get it working.
  - Then `aider`: `pip install aider-chat`; `aider --model openai/qwen2.5-coder:7b --openai-api-base http://127.0.0.1:<proxy>/v1 --yes --no-git --message "$(cat TASK.md)" solution.py`. 20 minutes max.
  - Then the repo's own loop: `calm solve` or arm C, base URL = proxy. Always works; weakest story. Say so in the bus.
- `bench_agent/trace_proxy.py`: **only if** Devin's `calm_proxy` v0 is not on `main` when you need it. ~80 lines: `aiohttp`/`http.server` passthrough of `/v1/chat/completions` (streaming included) to `http://127.0.0.1:11434`, writing the trace record exactly as the contract specifies to `--trace-dir`. If DV's lands later, switch to it and delete yours in the same commit.
- First real task through the proxy → `runs/<ts>_probe/` with `trace.jsonl`, `bodies/`, `result.json`. Commit and push **immediately**, even if the task failed; CF's headroom analysis needs the trace, not the solve. Then run the other 9 tasks of the probe with proxy = trace only (no levers). That is the `off` arm of the A/B too if the code path is identical (it is, if the only middleware is `trace`); record that decision.

Measure, don't guess, the one thing that decides I-2: from Ollama's `/v1` response, does `usage.prompt_tokens_details.cached_tokens` come back? The pilot said yes. If not, set `OLLAMA_DEBUG=1`, look at `prompt_eval_count` in `/api/chat`, and write what you found in the bus by 03:30 — CF's analysis has a fallback (prefix analysis on the bodies) but needs to know.

## Step 2 — the A/B (start as soon as DECISIONS.md is on `main`, target 06:30)

- `bench_agent/ab.py --tasks TASKS.json --arms off,on --proxy-flags "<from DECISIONS.md>"` runs every task under both arms, **interleaved** (`off,on,off,on…` per task, not all `off` then all `on`), so an Ollama slowdown does not land on one arm. Each arm writes its own run directory. `config.json` records git commit, agent, model, proxy flags, Ollama version.
- Concurrency 1. If a task exceeds 8 minutes it is `solved:false, reason:timeout` for that arm, never retried.
- Budget: 10 tasks × 2 arms × ~5 min ≈ 1h40. If by 05:30 fewer than 6 tasks are done, post `SLIP` and drop to 8 tasks.
- Commit each run directory as it completes. After the last one, run `python -m analysis.proxy_report runs/<off> runs/<on>` (CF's) and paste its two headline lines into `STATUS.md`. If CF's script is missing, write a 30-line one under `bench_agent/quick_report.py` and say so.
- Only if 06:30 arrives with ≥60 minutes to spare and DECISIONS.md lists I-6: the two-concurrent-sessions scenario. Otherwise `CUT`.

## Step 3 — final (by 08:15)

`STATUS.md` → "USER: read this first": (1) the headline number with the run directory; (2) anything blocked or that needs a permission; (3) the one command that shows the demo. `FINAL` entry in the bus. Push.

## Rules specific to you

- The Air is slow: ~20 tok/s. Never run two model jobs at once. Never start the old experiment driver tonight.
- Never edit `calm_proxy/**` or `analysis/proxy_*`. Post `REQ`s.
- Kill any process over its cap. A hung Ollama is restarted once (`ollama serve` or `brew services restart ollama`); if it dies again, record it and switch to CPU-only work: re-run CF's report on what exists, tighten `bench_agent/`, write the "USER" section.
- Commit at least every 30 minutes if anything changed. Push after every commit.
- `git pull --rebase` before every push; if a rebase conflicts on a file you own, keep yours; on a file you do not own, keep theirs and post a bus entry.

## Definition of done

- [ ] `bus/LC.md` with IDEAs, VERDICTs, hourly STATUS, FINAL
- [ ] `bench_agent/tasks/` (10 tasks) and `TASKS.json` on `main`
- [ ] one real agent running through a trace proxy; which one, in `STATUS.md`
- [ ] `runs/<ts>_probe/` on `main` by 03:30 (first task) and complete by 04:15
- [ ] A/B run directories `off` and `on` on `main` by 06:30
- [ ] `STATUS.md` with "USER: read this first" filled in

<!-- SHARED-CONTRACT:BEGIN — byte-identical in AGENT_LOCAL.md and AGENT_DEVIN.md. Change it in both or in neither. -->
## Shared contract (overnight pivot, Sun 2026-09-20)

### What changed and why

The research bet ("CALM-style coordination-free generation raises solve rate") is a **null result**: on the 39-task 1.5b run v2 − C = +0.026, McNemar p = 1.000; on the 7B pilot H2/H2b/H3/H6 fail. The harness is also slower (median TTFV 203 s vs 46 s) and sends 5× the prompt tokens. We stop defending that claim.

Two things in the same logs are real, exact, and reusable outside this harness:

1. **Content-addressing the verifier** avoids 64–75% of test executions (`results/cache.md`, 5,940 pairs, exact not sampled).
2. **Prefix-aligned prompts** hit the server's prompt cache ~94% of the time on Ollama (`results/final.md`, cached prompt tok / prompt tok).

The pivot is a **product** that gives those two savings, plus content-addressed dedup, to *any* agent that talks to a local model — Claude Code, aider, mini-swe-agent, OpenHands, the user's own scripts — without changing the agent:

> **CALM Proxy** — a content-addressed proxy for local model servers. Drop it between an agent and Ollama / vLLM / llama.cpp and it (a) shows where the agent is wasting prompt tokens and why the KV cache is missing, (b) memoizes identical requests and identical tool outputs, and (c) wraps test/tool commands so unchanged inputs are never re-executed.

Metric we sell: **prompt tokens actually computed** (prompt − cached), **test/tool executions avoided**, and **wall-clock** on a real agent loop, with **solve rate unchanged** (within CI). One of the levers becomes the headline at the T+1:00 decision (below); the rest ship as features with honestly reported numbers.

The CALM lineage survives as design, not as a claim: every store the proxy keeps is grow-only and keyed by content hash, so it is idempotent, replay-safe and needs no coordination. Say that once in the README and stop.

### Agents

| Code | Who | Where | Can generate tokens? | Role |
|---|---|---|---|---|
| **LC** | Claude Code (local) | Justin's MacBook Air, Ollama, `qwen2.5-coder:7b` (~20 tok/s single-stream — budget accordingly) | **Yes, the only one** | Agent-loop bench harness, every measurement, A/B runs, `STATUS.md` |
| **DV** | Devin | cloud, CPU | No | `calm_proxy/` package: server, trace, prefix-lint, memo, dedup, `calm-run`, tests |
| **CF** | Claude (cloud session, this file's author) | cloud, CPU | No | Research, `analysis/proxy_report.py` (trace → tables), decisions, PR review, README/pitch/demo page, final report |

Assumptions (if one is false, write it in your bus file and continue with the fallback named next to it; never stop):
- A1. DV and CF have no GPU and cannot reach Ollama. They never install torch/vllm. Fallback: develop against `calm_proxy/fake_upstream.py` (DV ships it first).
- A2. All three can push to `origin`. DV and CF open PRs and merge their own when tests pass. LC commits to `main`.
- A3. Agents cannot message each other directly. The bus (below) is the only channel besides `main`.
- A4. The user is asleep from ~T0 to ~08:00 ET. **No question is answered before then.** Any decision is made by the rules here, recorded, and reversible.

### Clock (America/New_York)

| Mark | Time | What |
|---|---|---|
| T0 | 02:45 | both agent files on `main`; everyone starts |
| T+0:30 | 03:15 | ideas window closes (≥2 IDEA entries per agent in the bus) |
| T+0:45 | 03:30 | DV: `calm_proxy` v0 (passthrough + trace) on `main`. LC: agent loop runs one task through *some* trace proxy |
| T+1:00 | 03:45 | **DECISION** (CF writes `bus/DECISIONS.md`): headline lever, from the probe numbers. If CF's decision is not on `main` by 03:55, LC decides by the same rule and writes it |
| T+2:00 | 04:45 | DV: all levers on `main` behind flags, tests green |
| T+3:45 | 06:30 | LC: A/B run directories on `main` (≥8 tasks, proxy off vs on) |
| T+4:45 | 07:30 | CF: `results/proxy.md`, README, demo page on `main`. **Feature freeze.** |
| T+5:30 | 08:15 | Docs-only from here. Every agent writes its final bus entry: what is on `main`, what was cut, what the user must check |
| 09:00 | deadline | |

If a mark slips by more than 30 minutes, the owner writes a `SLIP` entry naming the new time and what gets cut to make it. Cut order is at the bottom of this block.

### Ownership — never edit a path you do not own; reading and running anything is always fine

| Path | Owner |
|---|---|
| `calm_proxy/**` (server, middlewares, `calm-run` CLI, `fake_upstream.py`), `tests/proxy/**`, `pyproject.toml` entries for them | DV |
| `bench_agent/**` (task packaging, agent-loop launcher, A/B script, `trace_proxy.py` fallback), `runs/**` produced on the Mac, `STATUS.md` | LC |
| `analysis/proxy_*.py`, `tests/analysis/proxy_*`, `results/proxy*.md|json`, `docs/**`, `README.md`, `calm_proxy/demo/**` (page only), `bus/DECISIONS.md` | CF |
| `bus/LC.md`, `bus/DV.md`, `bus/CF.md` | that agent, append-only |
| Everything that already exists under `calm_coder/`, `analysis/` (old scripts), `runs/` (old dirs), `results/` (old files), `PREREG.md`, `CLAUDE.md`, `ARCHITECTURE.md` | **frozen**. Nobody edits. It is the record of the research phase and the README links to it. |

If a job needs a file in someone else's area, post `REQ` in your bus file and, if unanswered for 30 minutes, do the *minimum* edit in a clearly named PR titled `xo: <owner> <path>` and record it. Do not wait.

### Git protocol

- Before every push: `git fetch origin && git pull --rebase origin main` (or rebase your branch on `origin/main`). Bus files are per-agent so they never conflict.
- LC commits straight to `main`, small commits, at least every 30 minutes while anything changed.
- DV and CF: branches `devin/<thing>` and `cf/<thing>`, one PR per thing, opened as draft within 10 minutes, pushed often, **self-merged** when `pytest` is green for the paths touched. No PR waits for a human. No PR waits for another agent unless it touches their path.
- Never force-push `main`. Never rewrite, move or delete anything under `runs/`. Run directories are immutable and uniquely named (`runs/<YYYYMMDDTHHMMSSZ>_<name>/`).
- Commit `runs/` directories when finished (they are text). If one is over ~50 MB, commit `config.json`, `trace.jsonl` and `summary.md`, and say so in the bus.
- Every number in any document is produced by a script that reads a run directory, and the document names that directory. Nobody types a number by hand.

### The bus (the message channel)

`bus/<AGENT>.md`, append-only, your file only. One entry per event, newest at the bottom:

```
### 03:12 STATUS
mini-swe-agent runs against Ollama; first task traced to runs/20260920T071201Z_probe/.

### 03:14 IDEA-LC-1 tool-result dedup
<one paragraph: what, why it saves tokens, how measured, kill number>

### 03:31 VERDICT on IDEA-DV-2: adopt / adopt-if-cheap / drop — <one sentence why>

### 03:40 REQ-LC-1 (to DV, by 04:00): trace record needs `stream: bool`; I can't tell TTFT apart otherwise.

### 04:02 ACK REQ-LC-1: done in a1b2c3d.   |   ACK REQ-LC-1: declined — <why>

### 05:10 SLIP T+3:45 → 06:50; cutting the 2-agent affinity scenario.
```

Types: `STATUS`, `IDEA-<agent>-<n>`, `VERDICT`, `REQ-<agent>-<n>`, `ACK`, `SLIP`, `CUT`, `FINAL`. Read all three bus files (`git fetch` first) at the start of every task and at least every 30 minutes. Answer every `REQ` addressed to you within 30 minutes, even if the answer is "declined".

`bus/DECISIONS.md` is written only by CF (LC if CF misses the 03:55 fallback) and is the source of truth on *what ships*. If a bus entry and DECISIONS.md disagree, DECISIONS.md wins.

### Ideas window (T0 → T+0:30) and how ideas get adopted

Each agent posts at least two `IDEA` entries before 03:15, each with: the lever, the metric it moves, how the probe trace measures its headroom **before building it**, and a kill number. Each agent posts a `VERDICT` on every other agent's ideas by 03:35. CF folds the verdicts and the probe numbers into `bus/DECISIONS.md` by 03:45. After that, no new lever is started unless a `DECISION` entry adds it; anyone can keep posting `IDEA`s for the README's "what's next" section.

Seed ideas already on the table (post verdicts on these too):

| ID | Lever | Metric | Headroom measured by |
|---|---|---|---|
| I-1 `trace` | log every request/response with per-message hashes, usage, ttft, wall | none (it is the instrument) | — |
| I-2 `prefix-lint` | per session, longest common prefix with the previous request; first divergent message and byte; classify (timestamp / system-prompt churn / tool-result reorder / compaction) | prompt tokens *computed* = prompt − cached | (achievable prefix reuse) − (server-reported cached fraction) |
| I-3 `memo` | exact-request memo for temperature 0 / seeded requests, keyed by hash(model, messages, params) | requests and decode tokens avoided | share of requests in the trace whose key repeats |
| I-4 `dedup` | replace a tool-result message whose content hash already appeared earlier in the same session with a short stable reference (`[identical to result #k, sha256:…]`); stable across turns so the prefix stays aligned | prompt tokens | share of tool-result bytes in context that are exact repeats |
| I-5 `calm-run` | `calm-run -- pytest …`: hash(tracked tree, cmd, env subset) → cached stdout/stderr/exit; shim on PATH so the agent's shell hits it | test/tool executions avoided, wall | share of test commands in the trace re-run on an unchanged tree (the existing `results/cache.md` mechanism, on a real agent) |
| I-6 `affinity` | with several sessions on one llama.cpp/Ollama server, schedule back-to-back requests from the same session so its slot's KV prefix is not evicted | cached fraction under concurrency | two concurrent agent sessions, cached fraction with and without |

Decision rule at T+1:00: the headline lever is the one with the largest **measured headroom on the probe trace** among I-2..I-5, provided its kill number is not hit. I-1 always ships. I-6 ships only if the Mac can run two sessions at once and the A/B fits before 06:30. Ties break toward the lever that is already built.

### Trace record contract (so LC's fallback proxy and DV's proxy produce the same thing)

`trace.jsonl`, one object per upstream request, written when the response completes:

```json
{"ts": 1726815000.123, "session": "sha256-of-first-system-msg-or-header", "seq": 17,
 "model": "qwen2.5-coder:7b", "stream": true, "params": {"temperature": 0, "seed": null, "max_tokens": 1024},
 "messages": [{"role": "system", "sha": "…", "bytes": 4120}, {"role": "user", "sha": "…", "bytes": 300}, {"role": "tool", "sha": "…", "bytes": 5100, "tool_call_id": "…"}],
 "prompt_sha": "sha256 of the canonical JSON of messages",
 "usage": {"prompt_tokens": 6200, "cached_tokens": 5900, "completion_tokens": 210},
 "timing_ms": {"submit": 0, "first_token": 410, "done": 10900},
 "memo": {"hit": false, "key": "…"}, "dedup": {"replaced": 0, "bytes_saved": 0},
 "upstream_status": 200, "error": null}
```

Message bodies go in `bodies/<sha>.txt` next to the trace (content-addressed, written once). `cached_tokens` is `null` if the server does not report it. Additive changes only after 03:30.

### Agent-loop bench (what "a real agent" means tonight)

Tasks: ClassEval tasks re-packaged as tiny repos (`bench_agent/tasks/<id>/` with the skeleton, the hidden tests copied in as `tests/`, and a `TASK.md` saying "make the tests pass"). 8–10 tasks, chosen by seed 0 from the tasks the 7B model solved in `runs/20260919T163449Z_main` (so solve rate has signal) plus 2 it did not.

Agent: in this order of preference, first one working wins, 20 minutes each before falling to the next: (1) `mini-swe-agent` via litellm → `ollama/qwen2.5-coder:7b` with `OLLAMA_HOST`/`api_base` pointed at the proxy; (2) `aider --openai-api-base http://127.0.0.1:<proxy>/v1`; (3) the repo's own `calm solve` / arm C through the proxy. Record which in `STATUS.md`. Concurrency 1 (the Air cannot do more); one seed; `temperature 0`.

A/B: same tasks, same agent, `proxy=off` (direct to Ollama) vs `proxy=on` (all shipped levers). Report per task and pooled: solved, prompt tokens, cached tokens, computed tokens, completion tokens, requests, tool executions run vs memoized, wall. Solve rate difference is reported with its CI and *must* include 0 or favor `on`; if `on` loses solves, that lever is cut, not explained.

### Cut order (cut from the top when time runs out)

1. I-6 affinity (needs concurrency the laptop may not have)
2. Demo page (a README with tables and a 20-line `asciinema`-style transcript is enough)
3. I-3 memo (cheap but probably low headroom)
4. Second agent framework in the A/B
5. I-5 `calm-run` PATH shim (keep the CLI, drop the shim)

Never cut: trace, one A/B on one real agent, `results/proxy.md` generated by script, README that says what was measured and what was not.

### If the split is failing

- DV v0 not on `main` by 03:30: LC uses `bench_agent/trace_proxy.py` (minimal passthrough that writes the trace record above) and keeps going; DV's proxy must read/write the same format when it lands.
- LC has no agent loop running by 04:00: LC falls to option (3) and posts `SLIP`; CF adjusts the README framing to "on the harness's own loop".
- CF silent for 60 minutes: LC takes over `bus/DECISIONS.md` and the README; DV takes `results/proxy.md`.
- Someone edited a path they do not own: revert that commit, record it, do not patch around it.
- The Mac stops (sleep, Ollama died, permission prompt): LC records exactly what happened and what it needs in `STATUS.md` under **"USER: read this first"**, then does CPU-only work (analysis on existing traces). Do not retry a permission-gated action more than once.

### When the user wakes up

`STATUS.md` (LC) has a **"USER: read this first"** section at the top: 3 bullets max — is there a headline number, what is blocked, what to run to see the demo. `bus/DECISIONS.md` has the audit trail. `README.md` is the submission text.
<!-- SHARED-CONTRACT:END -->
