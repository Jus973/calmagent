# How to demo this

Two things are demoable: **the recorded results** (instant, cannot fail) and **the cache cliff
live** (~30 s, needs Ollama). Do the first as your spine and the second as your one live moment.

Do **not** try to run an agent live. A task takes 40 s to 8 minutes, the 7B model is erratic, and
the interesting effect only appears in long conversations. That is what the recorded runs are for.

---

## Pre-flight (do this before you present)

```bash
cd ~/Documents/github/calmagent
git pull
python -m bench_agent.demo | head -5      # should print instantly, no server needed
curl -s localhost:11434/api/version       # only needed for the live moment
```

If Ollama is not running:

```bash
OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 \
  OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

Those settings matter: the defaults give **4,096 tokens per sequence**, which agent prompts exceed
silently.

---

## The spine: one command, no model server

```bash
python -m bench_agent.demo
```

Runs in **0.1 s**. Prints all six sections — the cache cliff, the three killed levers, the
context-price fit, the cross-run finding, the replay distribution, the A/B — each naming the run
directory it came from. Nothing to fail on stage; it reads committed files.

Pipe it if the ANSI bold is noisy on a projector:

```bash
python -m bench_agent.demo | sed -e 's/\x1b\[[0-9;]*m//g'
```

Add `--verify` to re-derive the headline numbers from raw rows instead of the summaries. Worth
saying out loud that the flag exists; do not spend demo time on it.

---

## The live moment (~30 s): the cache cliff

This is the only thing worth running live, because a judge watching a 24-second wall clock and then
a 0.24-second one gets it instantly.

```bash
python -m bench_agent.probe_cache --quick --out /tmp/demo_$(date +%s)
```

What happens: a warm-up call (printed, explicitly not measured), then the **same 4,300-token
prompt twice** — once cold, once warm.

```
cold  ~24,000 ms
warm     ~240 ms
=> ~100x on prompt evaluation
```

**Say while it runs:** "same prompt, twice, and nothing in the OpenAI API tells the agent which one
it just got."

Safe to rehearse repeatedly: each invocation tags its prefix uniquely, so "cold" is genuinely cold
every time. (Without that, the second run of the day measures the first run's cache, prints
cold ≈ warm, and looks exactly like the lever not working.)

**Use a fresh `--out` each time.** Run directories are append-only; reusing one appends rows and
the printed summary will mix runs.

If it hangs or the model was evicted, the first number will include a ~7 GB weight load. The
warm-up prevents this — but if you see a 60 s cold number, that is what happened. Ctrl-C, re-run.

---

## If someone asks to see the proxy itself

```bash
# terminal 1 — proxy in front of Ollama
python -m bench_agent.trace_proxy --port 8999 --trace-dir /tmp/live --dedup

# terminal 2 — anything OpenAI-compatible, pointed at it
curl -s localhost:8999/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-coder:7b","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}'

# terminal 1 again — what it recorded
python -c "import json;[print(json.dumps({k:d[k] for k in ['seq','usage','timing_ms','prefix']})) for d in map(json.loads, open('/tmp/live/trace.jsonl'))]"
```

That is the whole product in three commands: an agent talks to it unchanged, and it writes a
content-addressed trace.

---

## Two traps that will bite you

**1. There are two proxies in this repo and only one implements the lever.**

| | implements | use it for |
|---|---|---|
| `bench_agent.trace_proxy` (`--dedup`) | trace **and dedup** | every result in `runs/`; anything you demo |
| `calm_proxy` (`--enable trace`) | trace only | tracing; it has `fake_upstream.py` for a server-free test |

`calm_proxy --enable dedup` used to start happily and do nothing — `replaced: 0` on every request,
indistinguishable from a null result. It now refuses with a message pointing at the right one. If
you are demoing the lever, use `bench_agent.trace_proxy`.

**2. Do not quote a single conversation's speedup.**

The range is 0.99x to 3.67x. Four of ten conversations show no effect. The number to say is the
pooled **2.12x on 37% fewer prompt tokens**, and then point at the four rows where it does nothing.
Someone will check.

---

## Recovery, by failure

| what breaks | do this |
|---|---|
| Ollama down / model evicted | Skip the live moment entirely. `python -m bench_agent.demo` needs no server and contains the same numbers. |
| Live probe prints cold ≈ warm | You reused a `--out` directory, or an older build without per-run prefix tags. Use a fresh `--out`; `git pull`. |
| Laptop asleep mid-demo | `caffeinate -dims &` before you start. |
| "Is this just prompt caching?" | See `docs/TALK.md` → Questions. Short version: compaction drops information and breaks the prefix; this is content-addressed, so the rewrite is identical every turn and the prefix stays aligned — and that stability is enforced by tests that caught a real bug. |

---

## The 30-second version, if the slot collapses

```bash
python -m bench_agent.demo | sed -e 's/\x1b\[[0-9;]*m//g' | head -14
```

> Your local model re-reads the agent's whole transcript — 26 seconds, versus a quarter second if
> the cache hits, and nothing in the API tells you which. We built the proxy that does. It told us
> three of our four planned optimisations were worth exactly zero, so we deleted them. The fourth
> gives 37% fewer prompt tokens and 2.1x less server time, solve rate unchanged — and nothing at
> all on short runs, which is the row we show too.
