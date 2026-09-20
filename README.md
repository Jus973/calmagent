# CALM Proxy

> A local model server will re-read your agent's whole transcript for 26 seconds, or reuse it in
> 0.25, and nothing in the OpenAI API tells you which one just happened. This is the instrument
> that tells you, and the two levers worth pulling once it has.

Drop it between any agent and Ollama / vLLM / llama.cpp. It records every request, shows where the
prompt cache is being missed and why, and shortens the prompts that are safe to shorten — without
changing the agent.

Built overnight at HackMIT 2026 on top of a research project whose main hypothesis came back null;
that project, its pre-registration and its failed hypotheses are still here, at the bottom.

## One command

```bash
python -m bench_agent.demo
```

Prints every number below out of the run directory that holds it. No model server, nothing to fail.

## What was measured

Each figure names the directory it came from. Nothing here is typed by hand.

### A prompt-cache hit is worth 100.9x — and the server will not tell you when you got one

`runs/20260920T063803Z_cache_probe/`, qwen2.5-coder:7b, a 4,317-token prompt:

| | prompt evaluation | per prompt token |
|---|---|---|
| cold prefix | 25,711 ms | 5.96 ms |
| warm prefix | **253 ms** | **0.059 ms** |

Two ways to throw it away, both measured in the same run: a **timestamp at the front of the system
prompt** costs 28,586 ms, and **one interleaved request** with a different prefix costs 27,327 ms
— with `OLLAMA_NUM_PARALLEL=1` there is a single KV slot, so an agent that alternates between two
prompt shapes evicts its own cache every time it switches.

**Ollama 0.21.2 reports no `cached_tokens` at all**, and `prompt_eval_count` does not shrink on a
hit. The contract this project started from assumed "prompt tokens computed = prompt − cached" was
readable. It is not, on this server. Wall clock is the only instrument there is, which is why every
number here is a time.

### On a well-behaved agent there is nothing left to win — and that is a result

`runs/20260920T074436Z_headroom/`, 42 real mini-swe-agent requests. Every lever was sized from the
trace *before* any of them was built:

| lever | headroom | verdict |
|---|---|---|
| I-2 prefix-lint | **0.0 s** — churn tokens: 0, all 40 divergences are `appended_only` | kill number hit |
| I-3 memo | **0.0 s** — 0 duplicate request keys of 42 | kill number hit |
| I-4 dedup | 102.3 s of uncached duplicate bytes | **shipped** |
| I-5 `calm-run` | 2 reruns of 5 test commands, ~20 ms each on this bench | shipped, uncaveated it would be dishonest |

mini-swe-agent only ever appends to its transcript: it never stamps it, reorders it or rewrites it.
Measured prefill across those 42 requests was 283.3 s, against 185.8 s for a perfect cache and
2,072.7 s for none — **the cache is already capturing 94.8% of what there is to capture**. A lint
that promised to win that back would be selling something already owned. It ships as a diagnostic
that prints a clean bill of health, and the 100.9x above is what it would find on an agent that
was not this well behaved.

### So why does the agent keep getting slower? Context is not free

`runs/20260920T072921Z_context_cost/`, 73 requests. The measured request time fitted against both
kinds of token, each with its own attention term (r² = 0.975):

```
done_ms = a*new + b*new*context + c*completion + d*completion*context
```

| | at 1,246 ctx | at 27,449 ctx | multiple |
|---|---|---|---|
| one prompt token | 7.36 ms | 16.04 ms | **2.18x** |
| one output token | 45.92 ms | 113.68 ms | **2.48x** |

Generation falls from ~22 tok/s to ~8.8 tok/s with nothing changed but the length of the transcript
it is appended to. No prefix cache addresses this: those tokens *are* cached, and are still being
attended to by every token generated.

Two independent checks that this is not an artefact of the fit, neither of which the fit was given:
`a` = 6.95 ms per prompt token against the cache probe's separately measured **5.96 ms**, and
`c` = 42.7 ms per output token, i.e. **23 tok/s**, against this model's known ~20 tok/s.

This is the argument for dedup, and it is not the argument for dedup that we started with. Dedup is
nearly worthless as a *prefill* saving, because the bytes it removes were mostly cached already.
It is worth something because a shorter context makes every remaining token of both kinds cheaper.

### Does it survive contact with a real agent?

<!-- AB-RESULTS -->

## What this does not claim

- **Not that we made the cache work better.** This agent's cache already works. What the proxy did
  was *prove* it, and measure the cliff waiting for an agent whose prompt is not append-only.
- **Not that dedup saves 620 s of prefill.** Deleting an already-cached token saves almost nothing.
  The defensible prefill figure is 102 s; the rest of dedup's case is the context curve above.
- **Not that `calm-run` pays off here.** The research phase measured 64–75% of *test executions*
  avoided (`results/cache.md`). A ClassEval test module runs in ~20 ms, so on this bench that
  converts to about 40 ms. A repo with a 30-second suite is where it pays, and this bench cannot
  show it.
- **Not that dedup improves solve rate.** See the A/B section: at temperature 0 the agent is
  deterministic given its prompt, so the first rewritten message forks the trajectory and the arms
  become different agents. Any solve-rate difference at n=10 is a coin flip.

## The CALM lineage

Every store the proxy keeps is grow-only and keyed by content hash, so it is idempotent,
replay-safe and needs no coordination — the dedup memory is a G-Set, and a rewrite is a pure
function of content, which is exactly why the same message is rewritten identically on every turn.
That stability is the whole lever: a rewrite that moved would invalidate the prefix behind it, and
at 5.96 ms per prompt token a single broken 17k-token prefix costs ~100 s, more than dedup saves
across a whole task. It is enforced by tests, not by convention
(`tests/bench_agent/test_trace_proxy_dedup.py`).

## Reproduce

```bash
# 1. what a cache hit is worth on your machine
python -m bench_agent.probe_cache --out runs/$(date -u +%Y%m%dT%H%M%SZ)_cache_probe

# 2. package the tasks and check every reference solution still passes
python -m bench_agent.make_tasks && python -m bench_agent.verify_tasks

# 3. the A/B, interleaved, one task at a time
python -m bench_agent.ab --arms off,on --agent mini-swe --label ab \
    --proxy-flags "--dedup --dedup-min-bytes 200"

# 4. read it
python -m bench_agent.quick_report runs/<ts>_ab_off runs/<ts>_ab_on
python -m bench_agent.probe_headroom runs/<ts>_ab_off
python -m bench_agent.context_cost runs/<ts>_ab_off
```

Serve with `OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=32768`. The default 4 parallel slots at
16k give **4,096 tokens per sequence**, which an agent prompt silently exceeds.

---

---

# Archive: the research phase

The hypothesis below — that a coordination-free, content-addressed store raises class-level solve
rate — was pre-registered in `PREREG.md` and came back **null**: v2 − C = +0.026, McNemar p = 1.000.
H1, H4 and H5 hold; H2, H2b, H3 and H6 fail. It is kept in full, including the failures, because
the proxy above is built out of what it measured.

# CALM Coder

> A shared file is a last-writer-wins register, so every merge is an ordering decision. A grow-only, hash-keyed set of definitions has no ordering decision to make. CALM tells you that is exactly the line between "needs coordination" and "doesn't."

Parsel-style per-method sample-and-verify, rebuilt on a coordination-free store, with a CALM analysis of where the single barrier actually is — evaluated on ClassEval against equal-budget whole-class sampling. HackMIT 2026. Everything runs on a laptop against a local model.

## The idea in one screen

N agents write **every method of a class concurrently**. Nothing they write ever overwrites anything:

- **Grow-only store (I1).** Exactly two writes, `add_def` and `add_outcome`, both idempotent set inserts. No update, delete, "keep the best", or "latest". A test greps the store for mutating calls.
- **Content-addressed (I2).** A definition's identity is the hash of its canonical form (docstrings, comments, formatting and local names erased). Calls to other *declared* methods are late-bound, so any fill of `get` composes with any fill of `put`. Calls to private helpers are frozen to the helper's hash, Unison-style, so two agents' `_norm` helpers never clash.
- **Monotone derivations (I3).** `complete`, `verified`, `done` use only ∃ / ∧ / transitive closure over immutable facts. Same facts in any order ⇒ same conclusions. Checked by replaying every real event log in 20 shuffled orders.

**What CALM buys, precisely.** The success exit is `∃ verified composition`, which is monotone, so any worker that sees it can announce completion with no coordination, and late emissions are inert. The *one* genuine barrier is the failure exit: "no solution within budget" is a ∀ over agents, so it must wait for everyone. The LLM layer is not confluent and doesn't need to be; the program's output (any verified composition) is.

**What it doesn't buy.** Methods are semantically coupled. If every sampled `select` returns tuples where the tests want lists, no amount of recombination helps. The experiment measures how much of "compositional generation is worse" is fragile assembly (which the store removes by construction) and how much is real coupling (which it can't).

## Demo (3 minutes)

| Left: git, one worktree per agent | Right: the store |
|---|---|
| `python -m calm_coder.demo.run_demo --git` | `python -m calm_coder.demo.run_demo` (live) or `--offline` (recorded) |
| 4 agents each write the whole `KVStore` into `kv.py`; merged with `-X theirs` in order 0→3 and 3→0; `diff` is non-empty. **Output depends on merge order.** | 4 agents × 4 methods stream into the grid, `conflicts 0`, duplicates collapse, root turns green; then 20 shuffled replays: **derived facts identical. Nothing to order.** |

Honest note: with qwen2.5-coder:7b both merge orders happened to pass the KVStore tests; the git side demonstrates order dependence, not a failing build.

## Use it on your own class

```bash
python -m calm_coder.cli solve path/to/skeleton.py --tests path/to/test_x.py --N 4 --live --out solved.py
```

Skeleton + unittest file in, a class that passes the tests out. Exit 0 on verified, 1 when the budget is exhausted.

The default path is latency-first, not phase-by-phase: a fill is stub-tested the moment it decodes, a
composition is tried as soon as every slot has one fill, `--width` compositions race (the first *to finish*
verified is the exit) and each one runs its test classes concurrently and stops at the first non-pass — a
composition needs every class to pass, so the rest would only add facts nobody is waiting for. Each slot
draws its next sample as soon as its previous one lands and stops once it has a passing fill, and generation
is cancelled at the `∃` exit (late emissions are inert, so nothing is lost).

Completions are streamed and each one **ends when the method it was asked for is complete** — including any
private helper that method calls, and never inside an unfinished `<think>` block or before the prefix parses.
The prompt asks for one method, but a 7B model keeps going: fences, prose, the neighbouring methods. Those
tokens are decoded at the same rate as the ones we need and nothing downstream reads them, so the request is
cut instead. The fill that lands is byte-identical to the one the full completion would have produced
(`tests/test_stream.py`); `--no-stop-at-fill` decodes to the end. Streamed samples also carry `ttft_ms`.

`--cache outcomes.jsonl` reuses test results across runs: a test
class's result depends only on the test module and the fills it can reach, both content hashes, so re-solving
a class — or one sharing helpers — skips the subprocess. Reused results are marked `reused:` in the log and
never counted as an execution. `--sequential` is the experiment's path (generate all N, then phase 1, then
search); the experiment itself takes `--test-width` and records `test_wall_ms` next to `test_ms`.

## Setup

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Serving is any OpenAI-compatible endpoint, configured by env only: `CALM_BASE_URL`, `CALM_MODEL`, `CALM_NO_N=1` if the server ignores `n`, `CALM_MAX_INFLIGHT`, `CALM_STREAM=1` to stream every sample (the early cut streams its own requests regardless; a server that answers a streamed request with a whole completion is read as one). On a Mac with Ollama, start the server with a per-sequence context big enough for the prompts (`OLLAMA_CONTEXT_LENGTH` is the **total** across parallel sequences; the defaults gave 512 tokens/sequence against ~615-token prompts):

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
export CALM_BASE_URL=http://localhost:11434/v1 CALM_MODEL=qwen2.5-coder:7b CALM_NO_N=1
```

On a GPU box: `vllm serve <model> --seed 0 --enable-prefix-caching` and change only `CALM_BASE_URL`.

## Reproduce the experiment

```bash
python -m calm_coder.bench.classeval --select          # already frozen: data/subset.json (78/100 eligible, 50 drawn)
python -m calm_coder.bench.experiment --arms a_greedy,c,calm --N 8 --seeds 0 --limit 30 --name main
python -m calm_coder.bench.confluence runs/<dir>        # H5
python -m calm_coder.bench.metrics runs/<dir>           # metrics.json + summary.md
python -m calm_coder.bench.charts runs/<dir>            # six charts, PNG + SVG
```

Latency is measured separately, one lever at a time, against a deterministic in-process server, because
what the levers change is scheduling and a real model's sampling variance would swamp it:

```bash
python -m calm_coder.bench.latency --repeat 3 --tail short   # results/latency.md
```

Arms: **CALM** (N fills per method → stub-context method tests → composition search), **C** (N whole-class samples, same oracle tests: the fair baseline), **C@tokens** (only the first C samples that fit in CALM's completion tokens: the headline comparison), **A** (holistic pass@1, unbiased from C's samples, plus a greedy sample). N ∈ {1,2,4,8} by nested subsampling of one N=8 generation. Agents never see tests; every arm is oracle-verified, so the headline number is coverage (as in *Large Language Monkeys*).

<!-- RESULTS -->

## v2: one store, several producers

v1 asks each agent for one method. v2 drops that restriction: a whole-class sample is decomposed
into its methods and every method lands in the same store, so pooling whole-class and per-method
producers costs nothing and loses nothing — each sample's own composition is still evaluated as a
whole, so v2 cannot do worse than best-of-N on the same samples. What the store then buys is where
the next tokens go: the slots with no passing candidate are the only ones resampled, conditioned on
the current best class and its failing test output. `wcr` is the fair baseline for that step —
same feedback, same information, whole class regenerated.

```bash
python -m calm_coder.bench.experiment --arms c --N 8 --seeds 0 --name c_base
python -m calm_coder.bench.experiment --arms v2,v2_pm,v2_f0,wcr --seeds 0 \
    --budget-from runs/<c_base dir> --feedback F1 --name v2
python -m analysis.final_report runs/<c_base dir> runs/<v2 dir>   # results/final.{md,json}
```

Every arm gets the same per-task decode budget: the tokens arm C spent at N=8, measured, not
estimated. Feedback is capped and scrubbed (F0 test names only, F1 + exception type and message,
F2 + the raising line); a test-source substring longer than 30 characters never reaches a prompt,
which is a test, not a promise. Hypotheses are preregistered in `PREREG.md` and the report marks
the failed ones as failed. `ARCHITECTURE.md` explains why the non-monotone reads (`best_class`,
`dead_slots`) live outside `store/derive.py`.

### Replay demo

```bash
python -m calm_coder.demo.export_web runs/<v2 dir> --task ClassEval_21 --arm v2
python -m http.server -d calm_coder/demo/web 8000
```

Best-of-N on the left, the store on the right, same recorded run, no model server. "Shuffle and
replay" replays the same events in a random order and prints the same store hash.

### Limitations

The benchmark's own tests are the verifier, so "solved" means "passes ClassEval's tests". Repair
reads failure messages (F1 by default), so it is not a black-box method. ClassEval is 2023-era and
may be in the model's training data. One model, one seed per arm unless stated, and the ablations
are pilot-sized.

## Layout

```
calm_coder/store/    defs normalize store derive materialize   (the CALM part; I1–I3 tested)
calm_coder/serve/    client prompts extract                    (any OpenAI-compatible server)
calm_coder/agents/   fill (phase 0) scheduler (phases 1–2)     (the only place ordering policy lives)
calm_coder/runner/   sandbox _run_tests tests                  (subprocess, temp cwd, per-class alarms)
calm_coder/bench/    classeval baselines experiment metrics charts confluence
calm_coder/viz/      live                                      (rich grid, replay, --replay-shuffled)
calm_coder/v2/       state decompose feedback producers harness       (v2: producers, repair)
analysis/            pool_existing dead_slots final_report
calm_coder/demo/     task test_task git_baseline run_demo recorded/ export_web web/
calm_coder/cli.py    calm solve
```

`CLAUDE.md` is the build spec; its §9 is the running decision log, including every deviation from the spec and why.

## Prior art

Parsel (Zelikman et al., NeurIPS 2023) is the closest work: decompose, sample per function, search combinations with tests. Per-function sample-and-verify is theirs, not ours. Also Hypothesis Search (Wang et al., 2023), FunCoder (Chen et al., 2024), Large Language Monkeys (Brown et al., 2024), AlphaCode (Li et al., 2022), ClassEval (Du et al., 2023) and its successors ClassEval-Pro and ClassEval-TDD (2026), Unison (content-addressed code), CALM (Hellerstein & Alvaro, CACM 2020; Ameloot, Neven & Van den Bussche, JACM 2013), Bloom^L (Conway et al., 2012): the store is a G-Set CRDT and the derivations are a monotone Datalog program.

**Contribution, without inflation:** (1) a CALM analysis of a multi-agent code-generation harness that locates the single barrier (the failure exit) and shows the success path is coordination-free; (2) a content-addressed grow-only store that makes N fills per method first-class and conflict-free and assembly mechanical, which separates fragile assembly from semantic coupling; (3) a confluence self-test on real logs; (4) an independence-vs-observed coupling metric.
