# CALM Coder v2 — final report

50 tasks in the union, seeds [0], 0 tasks excluded by the run.

## Solve rate (tasks are the unit of inference)

| arm | N | tasks | seeds/task | solve rate | 95% CI | interval | missing tasks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c | 8 | 50 | 1 | 0.30 | [0.19, 0.44] | wilson | - |
| a_greedy | 1 | 50 | 1 | 0.12 | [0.06, 0.24] | wilson | - |

## Paired comparisons (unit: task)

| comparison | tasks | diff | bootstrap 95% CI | McNemar tasks | arm only | baseline only | McNemar p | ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy-c | 50 | -0.180 | [-0.300, -0.080] | 50 | 0 | 9 | 0.004 | 0 |

## Systems

| arm | tasks | decode tok | prompt tok | cached prompt tok | requests | median TTFV ms | over budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy | 50 | 616 | 583 | - | 1.0 | 30288 | 0 |
| c | 50 | 4794 | 4665 | 2522 | 8.0 | 46281 | 0 |
