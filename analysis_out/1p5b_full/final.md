# CALM Coder v2 — final report

50 tasks in the union, seeds [0], 0 tasks excluded by the run.

## Solve rate (tasks are the unit of inference)

| arm | N | tasks | seeds/task | solve rate | 95% CI | interval | missing tasks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2_pm | 4 | 50 | 1 | 0.36 | [0.24, 0.50] | wilson | - |
| v2 | 4 | 50 | 1 | 0.32 | [0.21, 0.46] | wilson | - |
| c | 8 | 50 | 1 | 0.30 | [0.19, 0.44] | wilson | - |
| wcr | 4 | 50 | 1 | 0.22 | [0.13, 0.35] | wilson | - |
| a_greedy | 1 | 50 | 1 | 0.12 | [0.06, 0.24] | wilson | - |

## Paired comparisons (unit: task)

| comparison | tasks | diff | bootstrap 95% CI | McNemar tasks | arm only | baseline only | McNemar p | ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy-c | 50 | -0.180 | [-0.300, -0.080] | 50 | 0 | 9 | 0.004 | 0 |
| v2-c | 50 | +0.020 | [-0.100, +0.140] | 50 | 5 | 4 | 1.000 | 0 |
| v2_pm-c | 50 | +0.060 | [-0.060, +0.180] | 50 | 6 | 3 | 0.508 | 0 |
| wcr-c | 50 | -0.080 | [-0.160, -0.020] | 50 | 0 | 4 | 0.125 | 0 |

## Systems

| arm | tasks | decode tok | prompt tok | cached prompt tok | requests | median TTFV ms | over budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy | 50 | 616 | 583 | - | 1.0 | 30288 | 0 |
| c | 50 | 4794 | 4665 | 2522 | 8.0 | 46281 | 0 |
| v2 | 50 | 3823 | 22092 | 17657 | 9.3 | 234082 | 0 |
| v2_pm | 50 | 4516 | 24700 | 22635 | 9.8 | 375170 | 0 |
| wcr | 50 | 4380 | 10735 | 9565 | 4.3 | 53172 | 0 |

## Repair

| arm | feedback | repair rounds | solved by samples alone | solved | dead slots after sampling | dead slots at end |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | F1 | 2.14 | 9 | 16 | 1.94 | 1.64 |
| v2_pm | F1 | 1.04 | 9 | 18 | 1.46 | 1.30 |
| wcr | F1 | 0.82 | 10 | 11 | 1.68 | 1.64 |
