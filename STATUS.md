# Status

## Done

* v1 end to end: grow-only store, canonicalization, materialization, sandboxed verifier,
  per-method sampling, blame-driven recombination search, C / C+S / A baselines, metrics, charts,
  confluence check, post-hoc ceiling and recombination analyses.
* v2 (this branch):
  * producer-agnostic layer — `WholeClassProducer`, `PerMethodProducer`, `RepairProducer`,
    `WholeClassRepairProducer`, all writing through `add_def`/`add_outcome` only;
  * whole-class decomposition with the no-loss guarantee (each sample's own composition is
    evaluated first);
  * policy reads outside `derive.py`: verdict facts, deterministic `best_class`, `dead_slots`;
  * feedback levels F0/F1/F2 with a scrubber that keeps test source out of F0/F1;
  * prefix-aligned prompts, per-task warm-up, per-task decode budgets, `n`-batched requests;
  * arms `v2`, `v2_pm`, `v2_f0`, `v2_f2`, `wcr` in the experiment driver, budgets read from a real
    arm-C run;
  * analyses: `analysis/pool_existing.py`, `analysis/dead_slots.py`, `analysis/final_report.py`;
  * tests: prefix alignment, leak guard (F0/F1/F2), no-loss decomposition, replay determinism,
    budget enforcement, repair of a dead slot, report statistics.
* Docs: `ARCHITECTURE.md`, `PREREG.md`.

## Not done

* The v2 evaluation run itself (needs the local model server): a C run for budgets, then
  `--arms v2,v2_pm,v2_f0,wcr --budget-from <C run>`, then `analysis/final_report.py`.
* Phase 0 recon numbers: `analysis/pool_existing.py` and `analysis/dead_slots.py` are written but
  have not been run against `runs/` yet.
* Phase 5 replay-first demo page.

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
```

## Open questions

* `dead_slots` falls back to blame-based attribution for slots without their own test class; on
  tasks where class-level tests dominate this can target more slots than strictly necessary.
* Prompt-token accounting depends on the server reporting `prompt_tokens_details.cached_tokens`;
  Ollama does not, so cache counters are `null` there and the prefix claim rests on the prompt
  construction test rather than on measured cache hits.
