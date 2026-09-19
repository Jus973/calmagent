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

Serving is any OpenAI-compatible endpoint, configured by env only: `CALM_BASE_URL`, `CALM_MODEL`, `CALM_NO_N=1` if the server ignores `n`, `CALM_MAX_INFLIGHT`. On a Mac with Ollama, start the server with a per-sequence context big enough for the prompts (`OLLAMA_CONTEXT_LENGTH` is the **total** across parallel sequences; the defaults gave 512 tokens/sequence against ~615-token prompts):

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

Arms: **CALM** (N fills per method → stub-context method tests → composition search), **C** (N whole-class samples, same oracle tests: the fair baseline), **C@tokens** (only the first C samples that fit in CALM's completion tokens: the headline comparison), **A** (holistic pass@1, unbiased from C's samples, plus a greedy sample). N ∈ {1,2,4,8} by nested subsampling of one N=8 generation. Agents never see tests; every arm is oracle-verified, so the headline number is coverage (as in *Large Language Monkeys*).

<!-- RESULTS -->

## Layout

```
calm_coder/store/    defs normalize store derive materialize   (the CALM part; I1–I3 tested)
calm_coder/serve/    client prompts extract                    (any OpenAI-compatible server)
calm_coder/agents/   fill (phase 0) scheduler (phases 1–2)     (the only place ordering policy lives)
calm_coder/runner/   sandbox _run_tests tests                  (subprocess, temp cwd, per-class alarms)
calm_coder/bench/    classeval baselines experiment metrics charts confluence
calm_coder/viz/      live                                      (rich grid, replay, --replay-shuffled)
calm_coder/demo/     task test_task git_baseline run_demo recorded/
calm_coder/cli.py    calm solve
```

`CLAUDE.md` is the build spec; its §9 is the running decision log, including every deviation from the spec and why.

## Prior art

Parsel (Zelikman et al., NeurIPS 2023) is the closest work: decompose, sample per function, search combinations with tests. Per-function sample-and-verify is theirs, not ours. Also Hypothesis Search (Wang et al., 2023), FunCoder (Chen et al., 2024), Large Language Monkeys (Brown et al., 2024), AlphaCode (Li et al., 2022), ClassEval (Du et al., 2023) and its successors ClassEval-Pro and ClassEval-TDD (2026), Unison (content-addressed code), CALM (Hellerstein & Alvaro, CACM 2020; Ameloot, Neven & Van den Bussche, JACM 2013), Bloom^L (Conway et al., 2012): the store is a G-Set CRDT and the derivations are a monotone Datalog program.

**Contribution, without inflation:** (1) a CALM analysis of a multi-agent code-generation harness that locates the single barrier (the failure exit) and shows the success path is coordination-free; (2) a content-addressed grow-only store that makes N fills per method first-class and conflict-free and assembly mechanical, which separates fragile assembly from semantic coupling; (3) a confluence self-test on real logs; (4) an independence-vs-observed coupling metric.
