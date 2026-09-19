# Pre-registration — CALM Coder v2

Written before the v2 evaluation run. Anything decided after seeing v2 numbers is reported as
post-hoc, in its own section of `results/final.md`.

## Setup

* Benchmark: ClassEval, the pre-selected subset in `data/subset.json` (selection criteria and
  exclusions are in the run's `excluded.jsonl`; no task is ever special-cased in library code).
* Model: `qwen2.5-coder:7b`, temperature 0.8, top_p 0.95, served locally over an OpenAI-compatible
  endpoint. Model, server, and sampling parameters are recorded in each run's `config.json`.
* Seeds: the sampling seed of a request is
  `int(sha256(f"{run_seed}|{task_id}|{arm}|{slot}|{round}|{sample_idx}")[:8], 16)`.
  The run seed is part of the key, so two seeds of an arm are two samples of it rather than the
  same run logged twice; run seeds are recorded in `config.json` and in every result row.
* Unit of inference is the **task**: seeds are averaged within a task before any test.
* Budget: each task's decode-token budget equals the decode tokens arm C spent on that task at
  N=8 (`--budget-from <C run>`). Warm-up and repair tokens count against it.
* Feedback level for the headline arms: **F1**. F2 is reported separately and never mixed into a
  headline number.

## Arms

| arm | description |
| --- | --- |
| C | whole-class sampling, N samples, pass if any sample passes |
| C+S | C with the selection the harness already applies (best sample by tests) |
| CALM1 | v1: per-method sampling and recombination |
| V2 | whole-class samples pooled into the store, recombination, then targeted repair of dead slots |
| V2+PM | V2 with per-method samples pooled in as well |
| WCR | whole-class repair: same feedback, same budget, no recombination |
| V2-F0 | V2 with pass/fail-only feedback (ablation) |

## Hypotheses

* **H1** V2 solves more tasks than C at equal decode tokens.
* **H2** V2 solves more tasks than CALM1 at equal decode tokens.
* **H3** V2 solves more tasks than WCR at equal decode tokens and identical feedback — i.e.
  targeted repair plus recombination beats regenerating the whole class with the same information.
* **H4** Pooling is no-loss *up to carryable context*: on every task, V2's store contains a
  composition equal to every whole-class sample that would have passed on its own, whenever that
  sample's context is representable as a composition — imports, module constants and module-level
  helpers are carried into the lifted methods; a sample that rewrites the skeleton's constructor
  or adds class attributes is not, and is recorded in `ClassIngest.dropped` rather than counted.
  (Checked by construction and by test, not statistically.)
* **H5** Most unsolved tasks have exactly one dead slot, and interface/shared-state errors
  (AttributeError, KeyError, TypeError, NameError, IndexError) dominate value errors among them.
* **H6** Prefix alignment plus one warm-up request reduces prompt tokens computed per task relative
  to the v1 prompt layout, without changing solve rate.

## Analysis plan

* Solve rate per arm, tasks as the unit. Where a task's outcome is binary (one seed, or every
  seed agreeing) the interval is Wilson 95%; where seeds disagree a task's value is a fraction,
  which is not a Bernoulli trial, so the interval is a 10,000-resample percentile bootstrap over
  tasks. The report names which one it used.
* For each hypothesis: a paired bootstrap difference with 10,000 resamples over the tasks both
  arms ran, plus exact paired McNemar over the tasks where **both** arms are unanimous across
  seeds; tasks that are not are reported as `ambiguous`, never dropped silently. Significance at
  α = 0.05, two-sided; no correction across H1–H3 (reported as three pre-registered comparisons,
  not a family search).
* Every arm reports its own denominator: tasks run, seeds per task, tasks of the union it is
  missing, and the run's `excluded.jsonl`. Arms with different denominators are never compared
  on their headline rates alone.
* Solve-rate/token curves, cost and latency tables, dependency and failure-type breakdowns, and the
  dead-slot report (`analysis/dead_slots.py`).
* Everything is produced by `python -m analysis.final_report runs/<dir> ...` into
  `results/final.{md,json}`, with a copy under `<run>/analysis/` (run logs are never written to).
* Not yet implemented, and reported as such until they are: the H6 prompt-token comparison
  against the v1 layout (needs a v1 run logged with prompt/cached-prompt counts) and the
  solve-rate/token curve beyond the per-arm N sweep.

## Stopping rule

The evaluation ends when every task in the subset has completed every arm at its budget, or when
the wall-clock deadline for the hackathon is reached. A partially completed arm is reported with
its task count, never padded and never silently truncated.

## What would falsify the claim

* V2 ≤ WCR at equal tokens and equal feedback: recombination is not carrying the result.
* V2 ≤ C at equal tokens: pooling and repair do not pay for their overhead.
* A no-loss violation: a whole-class sample passes on its own but V2 fails the task.
