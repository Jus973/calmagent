# Status

## USER: read this first

- *(filled in before 08:15 — headline number, anything blocked, the one command to see the demo)*
- *(nothing is blocked as of 03:10; no permission prompt has been hit)*
- *(Ollama was restarted with `OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=32768`; your old
  4-slot/16k settings are not restored automatically)*

## Research phase result

The bet that CALM-style coordination-free generation raises solve rate is a **null**: on the
39-task 1.5b run v2 − C = +0.026 with McNemar p = 1.000, and on the 7B run H2/H2b/H3/H6 all fail.
Details and the pre-registration table with outcomes: `results/final.md`,
`runs/20260919T163449Z_main/summary.md`.

## Pivot

Overnight of Sun 2026-09-20 the work moved to **CALM Proxy**: a content-addressed proxy that sits
between any agent and a local model server. Instructions and the shared contract are in
`docs/overnight/`; the audit trail is `bus/` and `bus/DECISIONS.md`.

## Running log (overnight)

- **02:35** Ollama restarted single-slot at 32k context. The previous `NUM_PARALLEL=4` /
  `CONTEXT_LENGTH=16384` gives 4,096 tokens *per sequence*, which an agent prompt exceeds.
- **02:41** `runs/20260920T063803Z_cache_probe/`: a prefix-cache hit is worth **100.9×** on prompt
  evaluation (25,711 ms → 253 ms on a 4,317-token prompt). Ollama reports **no** `cached_tokens`,
  and `prompt_eval_count` does not shrink on a hit, so wall clock is the only instrument.
- **03:05** First real agent loop traced end to end: mini-swe-agent → `bench_agent/trace_proxy.py`
  → Ollama, on `bench_agent/tasks/ClassEval_7`.
- **03:12** Lever headroom measured on 42 real requests: I-2 prefix-lint **0 s** and I-3 memo
  **0 s** (both kill numbers hit — this agent is already well behaved and the server is capturing
  **94.8%** of the available cache saving); I-4 dedup **102 s**, the only survivor.
- **03:08** Dedup implemented behind `--dedup` with a stability test that caught a real prefix-
  breaking bug before it ever ran.

## How to reproduce tonight's numbers

```bash
python -m bench_agent.probe_cache --out runs/$(date -u +%Y%m%dT%H%M%SZ)_cache_probe
python -m bench_agent.make_tasks && python -m bench_agent.verify_tasks
python -m bench_agent.ab --arms off,on --agent mini-swe --label ab
python -m bench_agent.probe_headroom runs/<a trace dir>
```

---

# Archive (research phase)

## Done

* v1 end to end: grow-only store, canonicalization, materialization, sandboxed verifier,
  per-method sampling, blame-driven recombination search, C / C+S / A baselines, metrics, charts,
  confluence check, post-hoc ceiling and recombination analyses.
* v2 (this branch):
  * producer-agnostic layer — `WholeClassProducer`, `PerMethodProducer`, `RepairProducer`,
    `WholeClassRepairProducer`, all writing through `add_def`/`add_outcome` only;
  * whole-class decomposition, no-loss up to carryable context: each sample's own composition is
    evaluated first and the module context its methods read (imports, constants, module-level
    helpers) is carried into the lifted definitions; a rewritten constructor or class attributes
    cannot be carried and are reported in `ClassIngest.dropped`;
  * policy reads outside `derive.py`: verdict facts, deterministic `best_class`, `dead_slots`;
  * feedback levels F0/F1/F2 with a scrubber that keeps test source out of F0/F1;
  * prefix-aligned prompts, per-task warm-up carrying its family's system message, per-task decode
    budgets, `n`-batched requests, run-seeded sampling identities;
  * arms `v2`, `v2_pm`, `v2_f0`, `v2_f2`, `wcr` in the experiment driver, budgets read from a real
    arm-C run;
  * analyses: `analysis/pool_existing.py`, `analysis/dead_slots.py`, `analysis/final_report.py`;
  * tests: prefix alignment, leak guard (F0/F1/F2), no-loss decomposition, replay determinism,
    budget enforcement, repair of a dead slot, report statistics.
  * replay-first demo page (`calm_coder/demo/web/`) fed by `calm_coder.demo.export_web`.
* Docs: `ARCHITECTURE.md`, `PREREG.md`.
* A pilot run against a real model (qwen2.5-coder:7b via Ollama, 10 tasks, seed 0): arm C for the
  budgets, then `v2` and `wcr` under them, reported in `results/pilot.md` with the generated
  `results/final.md` / `results/final.json`.

* Heterogeneous producers (`serve/client.Fleet`): several models behind one `sample`, so the agents
  writing into a store are different models rather than clones. `Sample.model` / `Emission.model`
  carry provenance; identity stays the canonical hash, so two models writing the same method write
  one definition. `--models m1[@url],m2` on the experiment.
* Two analyses on recorded logs, no model and no subprocess: `analysis/cache_value.py` (what
  content-addressing saves the verifier) and `analysis/tail_waste.py` (how much of a real
  completion is tail the decode-side lever would cut).
* H5 confluence verified on every real run, not just the smoke run: 136 event logs, 34,763 events,
  20 shuffled orders each, 0 derived-fact differences.
* Metrics, charts and the pre-registration table for the 22-task 7B run
  (`runs/20260919T163449Z_main/summary.md`): H1, H4, H5 hold; H2, H2b, H3, H6 fail; F1 not triggered.

## Not done

* The preregistered evaluation at full scale: 50 tasks, 3 seeds, all five v2 arms. What exists is
  39 of 50 tasks at one seed on the 1.5b model, which is **underpowered by construction**: that
  model's holistic pass@1 is 10–12%, below the 30–55% band §3.6 requires, so 24 of 39 paired tasks
  are solved by neither arm and only 5 discordant pairs carry any information. v2 − C is +0.026,
  McNemar p = 1.000. Reported as a null, not as a loss.
* The mixed-model run (`--arms v2,wcr --models qwen2.5-coder:7b,llama3:8b`) is in flight; rows land
  per task, so whatever prefix finishes is what gets reported.
* `README.md` still has an empty `<!-- RESULTS -->` placeholder.
* PREREG H6 (prompt tokens vs the v1 layout) and the solve-rate/token curve beyond the per-arm N
  sweep are not implemented; the report does not claim them.
* Lint: `ruff` is not installed in this environment, so only `pytest` has been run.
* The demo's two recorded logs — the exporter and page are tested against a mock-model run, but
  the logs to ship are exported from the real run once it exists.

## How to run

```bash
# 1. baseline C, which also sets the per-task token budget
python -m calm_coder.bench.experiment --arms c --N 8 --seeds 0 --name c_base

# 2. v2 arms at equal decode tokens
python -m calm_coder.bench.experiment --arms v2,v2_pm,v2_f0,wcr --seeds 0 \
    --budget-from runs/<c_base dir> --feedback F1 --name v2

# 3. recon and report
python -m analysis.pool_existing runs/<earlier run>
python -m analysis.dead_slots runs/<earlier run>
python -m analysis.final_report runs/<c_base dir> runs/<v2 dir>

# 4. demo: export a recorded task, then serve the page (no model server involved)
python -m calm_coder.demo.export_web runs/<v2 dir> --task ClassEval_21 --arm v2
python -m http.server -d calm_coder/demo/web 8000
```

## Open questions

* A repair round's `repair_n` is split across dead slots (`repair_n // len(dead_slots)`, at least
  one sample each) so a targeted round costs about what a WCR round costs. When dead slots
  outnumber `repair_n`, the round asks for one sample per slot and therefore spends more than the
  baseline round; the budget still caps it, and the row records `samples_per_target`.
* `dead_slots` falls back to blame-based attribution for slots without their own test class; on
  tasks where class-level tests dominate this can target more slots than strictly necessary. When
  nothing names a slot at all the task is bucketed as unattributed and excluded from the
  "one dead slot" statistic.
* Prompt-token accounting depends on the server reporting `prompt_tokens_details.cached_tokens`.
  Ollama does report it (the pilot measured ~94% of v2's prompt tokens as cache hits); a server
  that omits it leaves the counters `null` and the prefix claim rests on the prompt construction
  test alone.
