# CALM Coder v2 — final report

50 tasks in the union, seeds [0], 0 tasks excluded by the run.

## Solve rate (tasks are the unit of inference)

| arm | N | tasks | seeds/task | solve rate | 95% CI | interval | missing tasks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2_pm | 4 | 48 | 1 | 0.38 | [0.25, 0.52] | wilson | ClassEval_85, ClassEval_97 |
| v2 | 4 | 50 | 1 | 0.32 | [0.21, 0.46] | wilson | - |
| c | 8 | 50 | 1 | 0.30 | [0.19, 0.44] | wilson | - |
| wcr | 4 | 47 | 1 | 0.21 | [0.12, 0.35] | wilson | ClassEval_85, ClassEval_86, ClassEval_97 |
| a_greedy | 1 | 50 | 1 | 0.12 | [0.06, 0.24] | wilson | - |

## Paired comparisons (unit: task)

| comparison | tasks | diff | bootstrap 95% CI | McNemar tasks | arm only | baseline only | McNemar p | ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy-c | 50 | -0.180 | [-0.300, -0.080] | 50 | 0 | 9 | 0.004 | 0 |
| v2-c | 50 | +0.020 | [-0.100, +0.140] | 50 | 5 | 4 | 1.000 | 0 |
| v2_pm-c | 48 | +0.062 | [-0.062, +0.188] | 48 | 6 | 3 | 0.508 | 0 |
| wcr-c | 47 | -0.085 | [-0.170, -0.021] | 47 | 0 | 4 | 0.125 | 0 |

## Systems

| arm | tasks | decode tok | prompt tok | cached prompt tok | requests | median TTFV ms | over budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy | 50 | 616 | 583 | - | 1.0 | 30288 | 0 |
| c | 50 | 4794 | 4665 | 2522 | 8.0 | 46281 | 0 |
| v2 | 50 | 3823 | 22092 | 17657 | 9.3 | 234082 | 0 |
| v2_pm | 48 | 4464 | 23691 | 21662 | 9.5 | 375170 | 0 |
| wcr | 47 | 4362 | 10792 | 9599 | 4.4 | 50538 | 0 |

## Repair

| arm | feedback | repair rounds | solved by samples alone | solved | dead slots after sampling | dead slots at end |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | F1 | 2.14 | 9 | 16 | 1.94 | 1.64 |
| v2_pm | F1 | 0.96 | 9 | 18 | 1.44 | 1.27 |
| wcr | F1 | 0.83 | 9 | 10 | 1.70 | 1.66 |
