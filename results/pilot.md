# Pilot run — qwen2.5-coder:7b, 10 ClassEval tasks, 1 seed

A pilot, not the preregistered evaluation: 10 of the 50 subset tasks, seed 0 only, on a CPU-only
box (8 cores, ~9 tok/s), so the intervals below are wide and no hypothesis in `PREREG.md` is
settled by them. Model `qwen2.5-coder:7b` served by Ollama at `http://localhost:11434/v1`,
temperature fixed for the run.

## Arms and budget

Baseline C ran first at N=1,2,4,8; its per-task N=8 decode-token spend is the budget every other
arm was given (`--budget-from`). Under that budget `v2` and `wcr` reached N=4 before the budget
capped them, so the comparison is "same decode tokens", not "same N". No arm went over budget.

```
python -m calm_coder.bench.experiment --arms c --N 8 --seeds 0 --limit 10 --name pilot_c
python -m calm_coder.bench.experiment --arms v2,wcr --N 8 --seeds 0 --limit 10 \
    --budget-from runs/<pilot_c> --name pilot_v2
python analysis/final_report.py runs/<pilot_c> runs/<pilot_v2>
python analysis/dead_slots.py runs/<pilot_v2> --arm v2 --N 4
```

## Result

| arm | N | solved | rate | 95% CI (Wilson) | decode tok/task | requests/task |
| --- | --- | --- | --- | --- | --- | --- |
| c | 8 | 6/10 | 0.60 | [0.31, 0.83] | 3797 | 8.0 |
| v2 | 4 | 8/10 | 0.80 | [0.49, 0.94] | 2567 | 4.5 |
| wcr | 4 | 6/10 | 0.60 | [0.31, 0.83] | 2825 | 3.2 |

Paired over the 10 common tasks: `v2 − c` = +0.200 (bootstrap CI [+0.000, +0.500], exact McNemar
p = 0.50, 2 tasks v2-only, 0 c-only); `wcr − c` = 0.000 (the same 6 tasks). The direction is what
the plan predicts and the magnitude is not significant at n = 10 — that is the expected power of a
pilot, not evidence for H1.

## Where the difference comes from

Both v2 and WCR solved 6 tasks by sampling alone — the same 6 as C. Repair is the whole gap:

| arm | repair rounds/task | solved by sampling | solved after repair | dead slots after sampling | dead slots at end |
| --- | --- | --- | --- | --- | --- |
| v2 | 0.90 | 6 | 8 | 1.10 | 0.20 |
| wcr | 0.40 | 6 | 6 | 1.10 | 1.10 |

Targeted repair closed 9 of the 11 dead slots; whole-class repair closed none of its 11, and spent
its budget re-emitting whole classes instead (fewer, larger requests: 3.2 vs 4.5 per task). On the
two tasks v2 still failed, `analysis/dead_slots.py` attributes one dead slot each — one
interface/state, one value error, none unattributed.

Prompt tokens are higher for v2 (10.6k vs 3.9k per task) because it issues more, smaller requests,
but 94% of them are prefix-cache hits that Ollama reports as `cached_tokens`, so the wall-clock and
cost effect is far smaller than the raw count suggests.

## What this does not show

* One seed, so nothing here separates run-to-run sampling variance from an arm effect.
* 10 tasks drawn in subset order, not the 50-task subset `PREREG.md` registers.
* `v2_pm`, `v2_f0`, `v2_f2` were not run, so the per-method and feedback-level ablations are open.
